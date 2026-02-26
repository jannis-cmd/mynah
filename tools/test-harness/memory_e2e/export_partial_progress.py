#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _psql_lines(container: str, db: str, user: str, query: str) -> list[str]:
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
    return [line.strip() for line in out.splitlines() if line.strip()]


def _norm_ts(ts_iso: str) -> str:
    dt = datetime.fromisoformat(ts_iso)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"timestamp must include timezone: {ts_iso}")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S+00")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export partial E2E ingest progress and remaining subsets.")
    parser.add_argument("--dataset-root", default="storage/test_data/memory_e2e")
    parser.add_argument("--report-path", default="reports/memory-e2e-partial-report.md")
    parser.add_argument("--progress-dir", default="storage/test_data/memory_e2e/progress")
    parser.add_argument("--postgres-container", default="mynah-postgres-1")
    parser.add_argument("--db-name", default="mynah")
    parser.add_argument("--db-user", default="mynah")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    progress_dir = Path(args.progress_dir).resolve()
    report_path = Path(args.report_path).resolve()

    transcripts_path = dataset_root / "voice_transcripts.jsonl"
    codex_root = dataset_root / "agent_history" / "sessions"
    manifest_path = dataset_root / "manifest.json"
    if not transcripts_path.exists():
        raise FileNotFoundError(f"missing transcripts: {transcripts_path}")
    if not codex_root.exists():
        raise FileNotFoundError(f"missing codex root: {codex_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact_rows = _psql_lines(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT source_type, processing_state, COUNT(*)
        FROM core.ingest_artifact
        GROUP BY source_type, processing_state
        ORDER BY source_type, processing_state;
        """,
    )
    artifact_counts: dict[tuple[str, str], int] = {}
    for row in artifact_rows:
        source_type, state, count = row.split("\t")
        artifact_counts[(source_type, state)] = int(count)

    ts_rows = _psql_lines(
        args.postgres_container,
        args.db_name,
        args.db_user,
        """
        SELECT source_ts::text, processing_state
        FROM core.ingest_artifact
        WHERE source_type = 'wearable_transcript' AND source_ts IS NOT NULL
        ORDER BY source_ts;
        """,
    )
    state_by_ts: dict[str, str] = {}
    for row in ts_rows:
        ts_text, state = row.split("\t")
        state_by_ts[ts_text] = state

    db_metric_queries = {
        "memory_note_count": "SELECT COUNT(*) FROM memory.note;",
        "health_sample_count": "SELECT COUNT(*) FROM health.sample;",
        "memory_health_link_count": "SELECT COUNT(*) FROM memory.health_link;",
        "compaction_attempt_count": "SELECT COUNT(*) FROM core.compaction_attempt;",
        "extraction_failure_count": "SELECT COUNT(*) FROM core.extraction_failure;",
    }
    db_metrics: dict[str, int] = {}
    for key, query in db_metric_queries.items():
        rows = _psql_lines(args.postgres_container, args.db_name, args.db_user, query)
        db_metrics[key] = int(rows[0]) if rows else 0

    transcripts = []
    with transcripts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            transcripts.append(json.loads(line))

    processed: list[dict] = []
    failed: list[dict] = []
    pending: list[dict] = []
    not_ingested: list[dict] = []
    for row in transcripts:
        key = _norm_ts(row["ts"])
        state = state_by_ts.get(key)
        if state == "processed":
            processed.append(row)
        elif state == "failed":
            failed.append(row)
        elif state == "pending":
            pending.append(row)
        else:
            not_ingested.append(row)

    remaining = failed + pending + not_ingested

    codex_files = sorted([p for p in codex_root.rglob("*.jsonl")])
    codex_processed_count = artifact_counts.get(("chat_export_codex", "processed"), 0)
    codex_failed_count = artifact_counts.get(("chat_export_codex", "failed"), 0)
    codex_pending_count = artifact_counts.get(("chat_export_codex", "pending"), 0)
    codex_remaining_count = max(0, len(codex_files) - codex_processed_count - codex_failed_count - codex_pending_count)

    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "processed_transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in processed) + ("\n" if processed else ""),
        encoding="utf-8",
    )
    (progress_dir / "failed_transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in failed) + ("\n" if failed else ""),
        encoding="utf-8",
    )
    (progress_dir / "pending_transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in pending) + ("\n" if pending else ""),
        encoding="utf-8",
    )
    (progress_dir / "remaining_transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in remaining) + ("\n" if remaining else ""),
        encoding="utf-8",
    )
    (progress_dir / "remaining_codex_files.txt").write_text(
        "\n".join(str(path.relative_to(codex_root)) for path in codex_files) + ("\n" if codex_files else ""),
        encoding="utf-8",
    )

    transcript_state_counts = Counter()
    transcript_state_counts["processed"] = len(processed)
    transcript_state_counts["failed"] = len(failed)
    transcript_state_counts["pending"] = len(pending)
    transcript_state_counts["not_ingested"] = len(not_ingested)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "transcript_total": len(transcripts),
        "transcript_state_counts": dict(transcript_state_counts),
        "remaining_transcript_count": len(remaining),
        "codex_total_files": len(codex_files),
        "codex_processed_artifacts": codex_processed_count,
        "codex_failed_artifacts": codex_failed_count,
        "codex_pending_artifacts": codex_pending_count,
        "codex_remaining_count": codex_remaining_count,
        "db_metrics": db_metrics,
        "artifact_counts": {
            f"{k[0]}:{k[1]}": v for k, v in sorted(artifact_counts.items(), key=lambda x: x[0])
        },
        "progress_files": {
            "processed_transcripts": str((progress_dir / "processed_transcripts.jsonl").relative_to(dataset_root)),
            "failed_transcripts": str((progress_dir / "failed_transcripts.jsonl").relative_to(dataset_root)),
            "pending_transcripts": str((progress_dir / "pending_transcripts.jsonl").relative_to(dataset_root)),
            "remaining_transcripts": str((progress_dir / "remaining_transcripts.jsonl").relative_to(dataset_root)),
            "remaining_codex_files": str((progress_dir / "remaining_codex_files.txt").relative_to(dataset_root)),
        },
    }
    (progress_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Memory E2E Partial Progress Report",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Persona: `{manifest.get('persona', {}).get('name', 'unknown')}`",
        "",
        "## Transcript Progress",
        "",
        "| State | Count |",
        "|---|---:|",
        f"| processed | {len(processed)} |",
        f"| failed | {len(failed)} |",
        f"| pending | {len(pending)} |",
        f"| not_ingested | {len(not_ingested)} |",
        f"| remaining_total | {len(remaining)} |",
        "",
        "## Codex History Progress",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| dataset_files_total | {len(codex_files)} |",
        f"| processed_artifacts | {codex_processed_count} |",
        f"| failed_artifacts | {codex_failed_count} |",
        f"| pending_artifacts | {codex_pending_count} |",
        f"| remaining_files_for_next_run | {codex_remaining_count} |",
        "",
        "## DB Snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| memory_note_count | {db_metrics['memory_note_count']} |",
        f"| memory_health_link_count | {db_metrics['memory_health_link_count']} |",
        f"| health_sample_count | {db_metrics['health_sample_count']} |",
        f"| compaction_attempt_count | {db_metrics['compaction_attempt_count']} |",
        f"| extraction_failure_count | {db_metrics['extraction_failure_count']} |",
        "",
        "## Resume Inputs",
        "",
        f"- Remaining transcripts: `{(progress_dir / 'remaining_transcripts.jsonl').relative_to(dataset_root)}`",
        f"- Failed transcripts: `{(progress_dir / 'failed_transcripts.jsonl').relative_to(dataset_root)}`",
        f"- Pending transcripts: `{(progress_dir / 'pending_transcripts.jsonl').relative_to(dataset_root)}`",
        f"- Remaining codex files: `{(progress_dir / 'remaining_codex_files.txt').relative_to(dataset_root)}`",
        f"- Summary JSON: `{(progress_dir / 'summary.json').relative_to(dataset_root)}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
