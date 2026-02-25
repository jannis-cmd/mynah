#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class IngestCounters:
    transcripts_total: int = 0
    transcripts_processed: int = 0
    transcripts_failed: int = 0
    codex_artifacts_total: int = 0
    codex_artifacts_processed: int = 0
    codex_artifacts_failed: int = 0
    health_samples_total: int = 0
    health_samples_inserted: int = 0


def _post_json(url: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} at {url}: {detail[:800]}") from exc


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(f"timezone required for timestamp: {ts}")
    return parsed.astimezone(timezone.utc)


def _chunk_text(lines: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = (current + "\n\n" + line).strip() if current else line
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(line) <= max_chars:
            current = line
        else:
            # Hard split for very long single lines.
            for i in range(0, len(line), max_chars):
                part = line[i : i + max_chars]
                if len(part) == max_chars:
                    chunks.append(part)
                else:
                    current = part
            if len(line) % max_chars == 0:
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _extract_codex_user_messages(jsonl_path: Path) -> list[tuple[datetime | None, str]]:
    rows: list[tuple[datetime | None, str]] = []
    for raw in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") != "message" or payload.get("role") != "user":
            continue
        content = payload.get("content")
        if not isinstance(content, list):
            continue
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "input_text":
                txt = str(part.get("text", "")).strip()
                if txt:
                    text_parts.append(txt)
        if not text_parts:
            continue
        ts: datetime | None = None
        ts_raw = obj.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                ts = _parse_iso(ts_raw.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                ts = None
        rows.append((ts, "\n".join(text_parts)))
    return rows


def _reset_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            TRUNCATE TABLE
              memory.health_link,
              memory.link,
              memory.note,
              decision.review,
              decision.entry,
              preference.fact,
              core.open_question,
              core.extraction_failure,
              core.compaction_attempt,
              core.report_artifact,
              core.transcript,
              core.audio_note,
              health.sample,
              core.artifact_meta,
              core.ingest_artifact
            RESTART IDENTITY CASCADE
            """
        )
    conn.commit()


def _ingest_health(api_base: str, hrv_csv: Path, counters: IngestCounters, chunk_size: int = 1200) -> None:
    rows = list(csv.DictReader(hrv_csv.read_text(encoding="utf-8").splitlines()))
    counters.health_samples_total = len(rows)
    for i in range(0, len(rows), chunk_size):
        batch = rows[i : i + chunk_size]
        payload = {
            "source": "memory_e2e_hrv_v1",
            "samples": [
                {
                    "ts": row["ts"],
                    "metric": row["metric"],
                    "value_num": float(row["value_num"]),
                    "unit": row["unit"],
                    "quality": int(row["quality"]),
                    "source": row["source"],
                }
                for row in batch
            ],
        }
        out = _post_json(f"{api_base}/ingest/health", payload, timeout=120)
        counters.health_samples_inserted += int(out.get("inserted", 0))


def _ingest_transcripts(api_base: str, transcript_jsonl: Path, counters: IngestCounters) -> None:
    lines = transcript_jsonl.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        if not raw.strip():
            continue
        row = json.loads(raw)
        counters.transcripts_total += 1
        ingest_payload = {
            "source_type": "wearable_transcript",
            "content": row["text"],
            "upload_ts": datetime.now(timezone.utc).isoformat(),
            "source_ts": row["ts"],
            "day_scope": False,
            "timezone": row.get("timezone", "UTC"),
            "caller": "memory_e2e_test",
        }
        ingest_out = _post_json(f"{api_base}/pipeline/artifacts/ingest", ingest_payload, timeout=60)
        artifact_id = ingest_out["artifact_id"]
        process_out = _post_json(
            f"{api_base}/pipeline/artifacts/process/{artifact_id}",
            {"caller": "memory_e2e_test"},
            timeout=300,
        )
        if process_out.get("status") == "ok":
            counters.transcripts_processed += 1
        else:
            counters.transcripts_failed += 1


def _ingest_codex_history(
    api_base: str,
    agent_history_root: Path,
    counters: IngestCounters,
    max_artifacts: int,
    max_chars_per_artifact: int = 5000,
) -> None:
    artifact_written = 0
    for jsonl_file in sorted(agent_history_root.rglob("*.jsonl")):
        rows = _extract_codex_user_messages(jsonl_file)
        if not rows:
            continue
        lines = []
        first_ts: datetime | None = None
        for ts, text in rows:
            ts_str = ts.isoformat() if ts else "unknown_ts"
            if first_ts is None and ts is not None:
                first_ts = ts
            lines.append(f"[{ts_str}] {text}")
        chunks = _chunk_text(lines, max_chars=max_chars_per_artifact)
        for idx, chunk in enumerate(chunks):
            if artifact_written >= max_artifacts:
                return
            counters.codex_artifacts_total += 1
            ingest_payload = {
                "source_type": "chat_export_codex",
                "content": chunk,
                "upload_ts": datetime.now(timezone.utc).isoformat(),
                "source_ts": first_ts.isoformat() if first_ts else None,
                "day_scope": False,
                "timezone": "UTC",
                "caller": "memory_e2e_test",
            }
            ingest_out = _post_json(f"{api_base}/pipeline/artifacts/ingest", ingest_payload, timeout=60)
            artifact_id = ingest_out["artifact_id"]
            process_out = _post_json(
                f"{api_base}/pipeline/artifacts/process/{artifact_id}",
                {"caller": f"memory_e2e_test_chunk_{idx}"},
                timeout=900,
            )
            artifact_written += 1
            if process_out.get("status") == "ok":
                counters.codex_artifacts_processed += 1
            else:
                counters.codex_artifacts_failed += 1


def _collect_db_stats(conn: psycopg.Connection) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT source_type, processing_state, COUNT(*) AS count
            FROM core.ingest_artifact
            GROUP BY source_type, processing_state
            ORDER BY source_type, processing_state
            """
        )
        out["artifact_state_counts"] = [dict(row) for row in cur.fetchall()]

        cur.execute("SELECT COUNT(*) AS count FROM memory.note")
        out["memory_note_count"] = int(cur.fetchone()["count"])

        cur.execute("SELECT ts_mode, COUNT(*) AS count FROM memory.note GROUP BY ts_mode ORDER BY ts_mode")
        out["ts_mode_counts"] = [dict(row) for row in cur.fetchall()]

        cur.execute("SELECT note_type, COUNT(*) AS count FROM memory.note GROUP BY note_type ORDER BY note_type")
        out["note_type_counts"] = [dict(row) for row in cur.fetchall()]

        cur.execute("SELECT COUNT(*) AS count FROM memory.health_link")
        out["memory_health_link_count"] = int(cur.fetchone()["count"])

        cur.execute("SELECT COUNT(*) AS count FROM core.extraction_failure")
        out["extraction_failure_count"] = int(cur.fetchone()["count"])

        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM core.compaction_attempt
            GROUP BY status
            ORDER BY status
            """
        )
        out["compaction_attempt_counts"] = [dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts, COUNT(*) AS count
            FROM health.sample
            WHERE metric = 'hrv_rmssd_ms'
            """
        )
        hrv = cur.fetchone()
        out["hrv_stats"] = {
            "count": int(hrv["count"] or 0),
            "min_ts": hrv["min_ts"].isoformat() if hrv["min_ts"] else None,
            "max_ts": hrv["max_ts"].isoformat() if hrv["max_ts"] else None,
        }
    return out


def _format_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "| Metric | Value |\n|---|---|\n| (none) | (none) |\n"
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def _build_report(
    dataset_root: Path,
    counters: IngestCounters,
    db_stats: dict[str, Any],
    pass_checks: list[tuple[str, bool, str]],
) -> str:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        ("transcripts_total", str(counters.transcripts_total)),
        ("transcripts_processed", str(counters.transcripts_processed)),
        ("transcripts_failed", str(counters.transcripts_failed)),
        ("codex_artifacts_total", str(counters.codex_artifacts_total)),
        ("codex_artifacts_processed", str(counters.codex_artifacts_processed)),
        ("codex_artifacts_failed", str(counters.codex_artifacts_failed)),
        ("hrv_samples_total", str(counters.health_samples_total)),
        ("hrv_samples_inserted", str(counters.health_samples_inserted)),
        ("memory_note_count", str(db_stats["memory_note_count"])),
        ("memory_health_link_count", str(db_stats["memory_health_link_count"])),
        ("extraction_failure_count", str(db_stats["extraction_failure_count"])),
    ]

    checks_lines = ["| Check | Pass | Detail |", "|---|---|---|"]
    for name, ok, detail in pass_checks:
        checks_lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")

    ts_modes = ", ".join(f"{row['ts_mode']}={row['count']}" for row in db_stats["ts_mode_counts"]) or "none"
    note_types = ", ".join(f"{row['note_type']}={row['count']}" for row in db_stats["note_type_counts"]) or "none"
    compaction = ", ".join(f"{row['status']}={row['count']}" for row in db_stats["compaction_attempt_counts"]) or "none"
    artifacts = ", ".join(
        f"{row['source_type']}:{row['processing_state']}={row['count']}" for row in db_stats["artifact_state_counts"]
    ) or "none"

    return (
        "# Memory E2E Ingest Report\n\n"
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"- Dataset root: `{dataset_root}`\n"
        f"- Transcript span: `{manifest.get('transcript_start_ts')}` -> `{manifest.get('transcript_end_ts')}`\n"
        f"- Persona: `{manifest.get('persona', {}).get('name', 'unknown')}`\n\n"
        "## Run Summary\n\n"
        + _format_table(rows)
        + "\n## Pass Criteria\n\n"
        + "\n".join(checks_lines)
        + "\n\n## DB Distributions\n\n"
        + _format_table(
            [
                ("artifact_states", artifacts),
                ("ts_mode_counts", ts_modes),
                ("note_type_counts", note_types),
                ("compaction_attempts", compaction),
                ("hrv_range", f"{db_stats['hrv_stats']['min_ts']} -> {db_stats['hrv_stats']['max_ts']}"),
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory E2E ingest loop and write report.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--db-dsn", default=None)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--max-codex-artifacts", type=int, default=20)
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    transcript_jsonl = dataset_root / "voice_transcripts.jsonl"
    hrv_csv = dataset_root / "hrv_samples.csv"
    agent_history_root = dataset_root / "agent_history" / "sessions"
    report_path = Path(args.report_path)
    if not transcript_jsonl.exists():
        raise FileNotFoundError(f"missing transcript dataset: {transcript_jsonl}")
    if not hrv_csv.exists():
        raise FileNotFoundError(f"missing hrv dataset: {hrv_csv}")
    if not agent_history_root.exists():
        raise FileNotFoundError(f"missing agent history dataset: {agent_history_root}")

    db_dsn = args.db_dsn or ""
    if not db_dsn:
        # Default inside mynah_agent container.
        db_dsn = "postgresql://mynah:mynah@postgres:5432/mynah"

    counters = IngestCounters()
    with psycopg.connect(db_dsn, autocommit=False) as conn:
        if args.reset_db:
            _reset_tables(conn)
        _ingest_health(args.api_base_url, hrv_csv, counters)
        _ingest_transcripts(args.api_base_url, transcript_jsonl, counters)
        _ingest_codex_history(
            args.api_base_url,
            agent_history_root,
            counters,
            max_artifacts=args.max_codex_artifacts,
        )
        db_stats = _collect_db_stats(conn)

    pass_checks: list[tuple[str, bool, str]] = []
    pass_checks.append(
        (
            "all_transcripts_processed",
            counters.transcripts_processed == counters.transcripts_total and counters.transcripts_total == 200,
            f"{counters.transcripts_processed}/{counters.transcripts_total}",
        )
    )
    pass_checks.append(
        (
            "no_transcript_failures",
            counters.transcripts_failed == 0,
            str(counters.transcripts_failed),
        )
    )
    pass_checks.append(
        (
            "hrv_insert_match",
            counters.health_samples_inserted == counters.health_samples_total,
            f"{counters.health_samples_inserted}/{counters.health_samples_total}",
        )
    )
    pass_checks.append(
        (
            "memory_notes_density",
            db_stats["memory_note_count"] >= math.floor(counters.transcripts_total * 2.0),
            f"{db_stats['memory_note_count']} >= {math.floor(counters.transcripts_total * 2.0)}",
        )
    )
    pass_checks.append(
        (
            "health_links_present",
            db_stats["memory_health_link_count"] > 0,
            str(db_stats["memory_health_link_count"]),
        )
    )
    pass_checks.append(
        (
            "extraction_failure_zero",
            db_stats["extraction_failure_count"] == 0,
            str(db_stats["extraction_failure_count"]),
        )
    )
    pass_checks.append(
        (
            "codex_history_processed",
            counters.codex_artifacts_processed > 0 and counters.codex_artifacts_failed == 0,
            f"processed={counters.codex_artifacts_processed}, failed={counters.codex_artifacts_failed}",
        )
    )

    report = _build_report(dataset_root=dataset_root, counters=counters, db_stats=db_stats, pass_checks=pass_checks)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "report_path": str(report_path),
        "pass_all": all(item[1] for item in pass_checks),
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in pass_checks],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
