#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JSON_RE = re.compile(r"\{[\s\S]*\}")
TIME_TOKEN_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b", re.IGNORECASE)


def _psql(container: str, db: str, user: str, query: str) -> list[str]:
    cmd = [
        "docker",
        "exec",
        container,
        "psql",
        "-U",
        user,
        "-d",
        db,
        "-t",
        "-A",
        "-F",
        "\t",
        "-c",
        query,
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="ignore")
    return [line.rstrip("\n") for line in out.splitlines() if line.strip()]


def _psql_json_rows(container: str, db: str, user: str, query: str) -> list[dict[str, Any]]:
    lines = _psql(container, db, user, query)
    out: list[dict[str, Any]] = []
    for line in lines:
        out.append(json.loads(line))
    return out


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = JSON_RE.search(text)
    if not match:
        raise ValueError("no json object found")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("parsed json is not object")
    return parsed


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _hint_confidence(hint: str, content_lc: str) -> tuple[str, str]:
    h = _norm(hint)
    if h in {"default", "today", "now"}:
        return "ok", "default/current"
    if h.startswith("yesterday"):
        if "yesterday" in content_lc or "last night" in content_lc:
            return "ok", "matched yesterday cue"
        return "weak", "yesterday hint but no clear yesterday cue in transcript"
    if h.startswith("tomorrow"):
        if "tomorrow" in content_lc:
            return "ok", "matched tomorrow cue"
        return "weak", "tomorrow hint but no clear tomorrow cue in transcript"
    if h in {"morning", "this morning"}:
        return ("ok", "matched morning cue") if "morning" in content_lc else ("weak", "morning hint without morning cue")
    if h in {"afternoon", "this afternoon"}:
        return ("ok", "matched afternoon cue") if "afternoon" in content_lc else ("weak", "afternoon hint without cue")
    if h in {"evening", "this evening"}:
        return ("ok", "matched evening cue") if "evening" in content_lc else ("weak", "evening hint without cue")
    if h in {"night", "tonight", "last night", "yesterday night", "tomorrow night"}:
        if "night" in content_lc or "tonight" in content_lc:
            return "ok", "matched night cue"
        return "weak", "night hint without cue"
    if h.startswith("at ") or h.startswith("yesterday at ") or h.startswith("tomorrow at "):
        return ("ok", "explicit time token found") if TIME_TOKEN_RE.search(content_lc) else ("weak", "time hint without explicit time token")
    return "weak", "unknown hint category"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit wearable transcript grouping/memory extraction quality.")
    parser.add_argument("--postgres-container", default="mynah-postgres-1")
    parser.add_argument("--db-name", default="mynah")
    parser.add_argument("--db-user", default="mynah")
    parser.add_argument("--report-path", default="reports/transcript-grouping-audit.md")
    parser.add_argument("--json-path", default="reports/transcript-grouping-audit.json")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    artifacts_rows = _psql_json_rows(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT json_build_object(
          'id', id,
          'source_ts', source_ts::text,
          'timezone', timezone,
          'content', content
        )::text
        FROM core.ingest_artifact
        WHERE source_type='wearable_transcript' AND processing_state='processed'
        ORDER BY source_ts;
        """,
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for row in artifacts_rows:
        art_id = str(row["id"])
        artifacts[art_id] = {
            "id": art_id,
            "source_ts": row["source_ts"],
            "timezone": row["timezone"],
            "content": row["content"],
        }

    notes_rows = _psql_json_rows(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT json_build_object(
          'source_artifact_id', source_artifact_id,
          'id', id::text,
          'ts', ts::text,
          'ts_mode', ts_mode,
          'note_type', note_type,
          'text', text
        )::text
        FROM memory.note
        WHERE source_artifact_id IN (
          SELECT id FROM core.ingest_artifact
          WHERE source_type='wearable_transcript' AND processing_state='processed'
        )
        ORDER BY source_artifact_id, id;
        """,
    )
    notes_by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in notes_rows:
        art_id = str(row["source_artifact_id"])
        notes_by_artifact[art_id].append(
            {
                "id": str(row["id"]),
                "ts": str(row["ts"]),
                "ts_mode": str(row["ts_mode"]),
                "note_type": str(row["note_type"]),
                "text": str(row["text"]).strip(),
            }
        )

    accepted_rows = _psql_json_rows(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT DISTINCT ON (artifact_id) json_build_object(
          'artifact_id', artifact_id,
          'attempt', attempt,
          'output_text', output_text
        )::text
        FROM core.compaction_attempt
        WHERE status='accepted'
        ORDER BY artifact_id, attempt DESC;
        """,
    )
    accepted_by_artifact: dict[str, dict[str, Any]] = {}
    for row in accepted_rows:
        art_id = str(row["artifact_id"])
        accepted_by_artifact[art_id] = {"attempt": int(row["attempt"]), "output_text": str(row["output_text"])}

    rejected_rows = _psql_json_rows(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT json_build_object(
          'artifact_id', artifact_id,
          'attempt', attempt,
          'error_text', LEFT(COALESCE(error_text,''), 260)
        )::text
        FROM core.compaction_attempt
        WHERE status='rejected'
        ORDER BY artifact_id, attempt;
        """,
    )

    failed_pending_rows = _psql_json_rows(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT json_build_object(
          'id', id,
          'source_ts', source_ts::text,
          'processing_state', processing_state
        )::text
        FROM core.ingest_artifact
        WHERE source_type='wearable_transcript' AND processing_state IN ('failed','pending')
        ORDER BY source_ts;
        """,
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_artifact_count": len(artifacts),
        "artifact_with_accepted_compaction": 0,
        "total_notes": 0,
        "total_group_items": 0,
        "hint_counts": Counter(),
        "hint_quality_counts": Counter(),
        "ts_mode_counts": Counter(),
        "note_type_counts": Counter(),
        "text_alignment_mismatches": 0,
        "default_only_artifacts_with_temporal_cues": 0,
        "default_only_artifact_ids": [],
        "weak_hint_examples": [],
        "alignment_mismatch_examples": [],
        "failed_or_pending_artifacts": [],
        "rejected_error_categories": Counter(),
    }

    temporal_cue_re = re.compile(r"\b(yesterday|tomorrow|morning|afternoon|evening|night|tonight|am|pm)\b", re.IGNORECASE)

    for art_id, artifact in artifacts.items():
        notes = notes_by_artifact.get(art_id, [])
        summary["total_notes"] += len(notes)
        for note in notes:
            summary["ts_mode_counts"][note["ts_mode"]] += 1
            summary["note_type_counts"][note["note_type"]] += 1

        accepted = accepted_by_artifact.get(art_id)
        if not accepted:
            continue
        summary["artifact_with_accepted_compaction"] += 1
        content_lc = artifact["content"].lower()
        parsed = _parse_json_object(accepted["output_text"])
        groups = parsed.get("groups", [])
        if not isinstance(groups, list):
            groups = []

        item_texts_from_groups: list[str] = []
        hint_set = set()
        for group in groups:
            if not isinstance(group, dict):
                continue
            hint = str(group.get("hint", "default")).strip()
            hint_set.add(_norm(hint))
            summary["hint_counts"][_norm(hint)] += 1
            quality, reason = _hint_confidence(hint, content_lc)
            summary["hint_quality_counts"][quality] += 1
            if quality == "weak" and len(summary["weak_hint_examples"]) < args.max_examples:
                summary["weak_hint_examples"].append(
                    {
                        "artifact_id": art_id,
                        "source_ts": artifact["source_ts"],
                        "hint": hint,
                        "reason": reason,
                        "content_excerpt": artifact["content"][:220].replace("\n", " "),
                    }
                )
            items = group.get("items", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                else:
                    text = str(item).strip()
                if text:
                    item_texts_from_groups.append(text)

        summary["total_group_items"] += len(item_texts_from_groups)
        if hint_set <= {"default"} and temporal_cue_re.search(content_lc):
            summary["default_only_artifacts_with_temporal_cues"] += 1
            if len(summary["default_only_artifact_ids"]) < args.max_examples:
                summary["default_only_artifact_ids"].append(art_id)

        notes_counter = Counter([n["text"] for n in notes])
        groups_counter = Counter(item_texts_from_groups)
        missing = groups_counter - notes_counter
        extra = notes_counter - groups_counter
        if missing or extra:
            summary["text_alignment_mismatches"] += 1
            if len(summary["alignment_mismatch_examples"]) < args.max_examples:
                summary["alignment_mismatch_examples"].append(
                    {
                        "artifact_id": art_id,
                        "source_ts": artifact["source_ts"],
                        "missing_group_items": list(missing.items())[:6],
                        "extra_db_items": list(extra.items())[:6],
                    }
                )

    for row in rejected_rows:
        e = str(row["error_text"]).lower()
        if "unsupported hint" in e:
            summary["rejected_error_categories"]["unsupported_hint"] += 1
        elif "expecting ',' delimiter" in e or "extra data" in e or "json" in e:
            summary["rejected_error_categories"]["json_parse"] += 1
        else:
            summary["rejected_error_categories"]["other"] += 1

    for row in failed_pending_rows:
        summary["failed_or_pending_artifacts"].append(
            {"artifact_id": str(row["id"]), "source_ts": str(row["source_ts"]), "state": str(row["processing_state"])}
        )

    # Convert counters for JSON serialization.
    for key in ("hint_counts", "hint_quality_counts", "ts_mode_counts", "note_type_counts", "rejected_error_categories"):
        summary[key] = dict(summary[key])

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Transcript Grouping Audit",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Processed wearable transcripts audited: `{summary['processed_artifact_count']}`",
        f"- Artifacts with accepted compaction output: `{summary['artifact_with_accepted_compaction']}`",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_notes (DB) | {summary['total_notes']} |",
        f"| total_group_items (LLM output) | {summary['total_group_items']} |",
        f"| text_alignment_mismatches | {summary['text_alignment_mismatches']} |",
        f"| default_only_artifacts_with_temporal_cues | {summary['default_only_artifacts_with_temporal_cues']} |",
        "",
        "## Distributions",
        "",
        "### ts_mode",
        "",
        "| ts_mode | count |",
        "|---|---:|",
    ]
    for k, v in sorted(summary["ts_mode_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "### note_type", "", "| note_type | count |", "|---|---:|"])
    for k, v in sorted(summary["note_type_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "### hints", "", "| hint | count |", "|---|---:|"])
    for k, v in sorted(summary["hint_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "### hint_quality", "", "| quality | count |", "|---|---:|"])
    for k, v in sorted(summary["hint_quality_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Failures and Risks",
            "",
            "### Failed/Pending Artifacts",
            "",
            "| artifact_id | source_ts | state |",
            "|---|---|---|",
        ]
    )
    for row in summary["failed_or_pending_artifacts"]:
        lines.append(f"| {row['artifact_id']} | {row['source_ts']} | {row['state']} |")

    lines.extend(
        [
            "",
            "### Rejected Attempt Error Categories",
            "",
            "| category | count |",
            "|---|---:|",
        ]
    )
    for k, v in sorted(summary["rejected_error_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")

    lines.extend(["", "### Weak Hint Examples", ""])
    for ex in summary["weak_hint_examples"][: args.max_examples]:
        lines.append(
            f"- `{ex['artifact_id']}` @ `{ex['source_ts']}` hint=`{ex['hint']}`: {ex['reason']} | excerpt: {ex['content_excerpt']}"
        )

    lines.extend(["", "### Alignment Mismatch Examples", ""])
    if not summary["alignment_mismatch_examples"]:
        lines.append("- none")
    else:
        for ex in summary["alignment_mismatch_examples"][: args.max_examples]:
            lines.append(
                f"- `{ex['artifact_id']}` @ `{ex['source_ts']}` missing={ex['missing_group_items']} extra={ex['extra_db_items']}"
            )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
