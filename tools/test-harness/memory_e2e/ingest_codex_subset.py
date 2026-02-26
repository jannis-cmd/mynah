#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
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
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_iso(ts: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
            return None
        return parsed.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _extract_user_lines(jsonl_path: Path) -> tuple[list[str], datetime | None]:
    lines: list[str] = []
    earliest: datetime | None = None
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
        text_parts = []
        for part in payload.get("content", []):
            if isinstance(part, dict) and part.get("type") == "input_text":
                txt = str(part.get("text", "")).strip()
                if txt:
                    text_parts.append(txt)
        if not text_parts:
            continue
        ts = _parse_iso(str(obj.get("timestamp", "")))
        if ts and (earliest is None or ts < earliest):
            earliest = ts
        ts_token = ts.isoformat() if ts else "unknown_ts"
        lines.append(f"[{ts_token}] " + "\n".join(text_parts))
    return lines, earliest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a subset of codex chat-export files.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=7000)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--remaining-file-list", default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    codex_root = dataset_root / "agent_history" / "sessions"
    if not codex_root.exists():
        raise FileNotFoundError(f"codex sessions root missing: {codex_root}")

    if args.remaining_file_list:
        remaining = Path(args.remaining_file_list)
        if remaining.exists():
            rel_files = [line.strip() for line in remaining.read_text(encoding="utf-8").splitlines() if line.strip()]
            file_paths = [codex_root / rel for rel in rel_files if (codex_root / rel).exists()]
        else:
            file_paths = sorted(codex_root.rglob("*.jsonl"))
    else:
        file_paths = sorted(codex_root.rglob("*.jsonl"))

    selected = file_paths[: args.count]
    if not selected:
        raise RuntimeError("no codex files selected")

    before = _get_json(f"{args.api_base_url}/status")
    results: list[dict[str, Any]] = []
    for idx, path in enumerate(selected, start=1):
        rel = str(path.relative_to(codex_root))
        lines, earliest = _extract_user_lines(path)
        if not lines:
            results.append(
                {
                    "file": rel,
                    "status": "skipped_no_user_messages",
                    "artifact_id": None,
                    "notes_created": 0,
                    "error": None,
                }
            )
            continue
        content = "\n\n".join(lines)
        if len(content) > args.max_chars:
            content = content[: args.max_chars]
        ingest_payload = {
            "source_type": "chat_export_codex",
            "content": content,
            "upload_ts": datetime.now(timezone.utc).isoformat(),
            "source_ts": earliest.isoformat() if earliest else None,
            "day_scope": False,
            "timezone": "UTC",
            "caller": f"codex_subset_{idx}",
        }
        try:
            ingest = _post_json(f"{args.api_base_url}/pipeline/artifacts/ingest", ingest_payload, timeout=60)
            artifact_id = ingest["artifact_id"]
            process = _post_json(
                f"{args.api_base_url}/pipeline/artifacts/process/{artifact_id}",
                {"caller": f"codex_subset_{idx}"},
                timeout=900,
            )
            results.append(
                {
                    "file": rel,
                    "status": process.get("status", "unknown"),
                    "artifact_id": artifact_id,
                    "notes_created": int(process.get("notes_created", 0)) if isinstance(process, dict) else 0,
                    "links_created": int(process.get("links_created", 0)) if isinstance(process, dict) else 0,
                    "error": process.get("reason") if isinstance(process, dict) else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "file": rel,
                    "status": "exception",
                    "artifact_id": None,
                    "notes_created": 0,
                    "links_created": 0,
                    "error": str(exc),
                }
            )

    after = _get_json(f"{args.api_base_url}/status")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = sum(1 for r in results if r["status"] not in {"ok", "skipped_no_user_messages"})
    skipped_count = sum(1 for r in results if r["status"] == "skipped_no_user_messages")

    report_lines = [
        "# Codex Quarter Ingest Report",
        "",
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Selected files: `{len(selected)}`",
        f"- Outcomes: ok=`{ok_count}`, failed=`{fail_count}`, skipped=`{skipped_count}`",
        "",
        "## Per File Result",
        "",
        "| file | status | artifact_id | notes_created | error |",
        "|---|---|---|---:|---|",
    ]
    for row in results:
        report_lines.append(
            f"| {row['file']} | {row['status']} | {row['artifact_id'] or ''} | {row.get('notes_created', 0)} | {(row.get('error') or '').replace('|','/')} |"
        )

    report_lines.extend(
        [
            "",
            "## DB Delta (status endpoint)",
            "",
            "| metric | before | after | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("artifact_count", "memory_note_count", "memory_health_link_count", "compaction_attempt_count"):
        b = int(before.get(key, 0))
        a = int(after.get(key, 0))
        report_lines.append(f"| {key} | {b} | {a} | {a - b} |")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    out = {
        "selected_files": len(selected),
        "ok": ok_count,
        "failed": fail_count,
        "skipped": skipped_count,
        "report_path": str(report_path),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
