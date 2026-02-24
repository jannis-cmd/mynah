#!/usr/bin/env sh
set -eu

docker compose exec -T mynah_ui python - <<'PY'
import base64
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = f"qlt_{int(time.time())}"
CALLER = RUN_ID
THEMES = ["sleep", "exercise", "nutrition", "stress", "focus"]
COUNT_PER_THEME = 20
TOTAL = len(THEMES) * COUNT_PER_THEME
THRESHOLDS = {
    "precision": 0.75,
    "recall": 0.70,
    "citation_validity": 0.98,
    "false_insight_rate": 0.10,
    "stale_memory_leakage_rate": 0.00,
}


def post(url: str, payload: dict, timeout: int = 20) -> dict:
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
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> {exc.code}: {body}") from exc


def get(url: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} -> {exc.code}: {body}") from exc


started = time.perf_counter()
memory_ids = []
theme_expected = {theme: set() for theme in THEMES}

today = datetime.now(timezone.utc).date().isoformat()
audio_bytes = base64.b64encode(b"RIFF_QUALITY_EVAL").decode("ascii")

for theme in THEMES:
    for idx in range(COUNT_PER_THEME):
        note_id = f"{RUN_ID}_{theme}_{idx:03d}"
        theme_expected[theme].add(note_id)
        transcript = (
            f"{RUN_ID} theme:{theme} note:{note_id} "
            f"I tracked {theme} today with consistent routine and reflective journaling."
        )
        ingest_payload = {
            "note_id": note_id,
            "device_id": "fixture_wearable_01",
            "start_ts": f"{today}T12:00:00+00:00",
            "end_ts": f"{today}T12:00:08+00:00",
            "audio_b64": audio_bytes,
            "transcript_hint": transcript,
            "source": "quality_eval",
        }
        post("http://mynahd:8001/ingest/audio", ingest_payload, timeout=15)
        transcribed = post(
            "http://mynah_agent:8002/pipeline/audio/transcribe",
            {"audio_id": note_id, "caller": CALLER, "force": True},
            timeout=20,
        )
        if transcribed.get("memory_id"):
            memory_ids.append(transcribed["memory_id"])

total_tp = 0
total_fp = 0
total_expected = TOTAL
total_returned = 0
total_stale_hits = 0
per_theme = {}

for theme in THEMES:
    query = f"{RUN_ID} theme:{theme}"
    result = post(
        "http://mynah_agent:8002/tools/memory_search",
        {"query": query, "limit": 100, "verified_only": True},
        timeout=15,
    )
    entries = result.get("entries", [])
    total_returned += len(entries)
    returned_ids = set()
    tp = 0
    stale_hits = 0
    for entry in entries:
        summary = entry.get("summary", "")
        if entry.get("stale") is True:
            stale_hits += 1
        if RUN_ID in summary and f"theme:{theme}" in summary:
            tp += 1
            marker = "note:"
            if marker in summary:
                returned_ids.add(summary.split(marker, 1)[1].split()[0].strip())
    fp = max(0, len(entries) - tp)
    fn = max(0, COUNT_PER_THEME - tp)
    total_tp += tp
    total_fp += fp
    total_stale_hits += stale_hits
    per_theme[theme] = {
        "expected": COUNT_PER_THEME,
        "returned": len(entries),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "stale_hits": stale_hits,
        "coverage_ids": len(returned_ids & theme_expected[theme]),
    }

verified_count = 0
for memory_id in sorted(set(memory_ids)):
    verify = get(f"http://mynah_agent:8002/tools/memory_verify/{memory_id}", timeout=10)
    if verify.get("verified") is True:
        verified_count += 1

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / total_expected if total_expected > 0 else 0.0
citation_validity = verified_count / len(set(memory_ids)) if memory_ids else 0.0
false_insight_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
stale_memory_leakage_rate = total_stale_hits / total_returned if total_returned > 0 else 0.0
runtime_sec = round(time.perf_counter() - started, 2)

summary = {
    "run_id": RUN_ID,
    "total_transcripts": TOTAL,
    "themes": per_theme,
    "metrics": {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "citation_validity": round(citation_validity, 4),
        "false_insight_rate": round(false_insight_rate, 4),
        "stale_memory_leakage_rate": round(stale_memory_leakage_rate, 4),
        "runtime_sec": runtime_sec,
    },
    "thresholds": THRESHOLDS,
}

report_dir = Path("/home/appuser/data/artifacts/reports/quality")
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / f"{RUN_ID}.json"
report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"quality_report={report_path}")

failed = (
    precision < THRESHOLDS["precision"]
    or recall < THRESHOLDS["recall"]
    or citation_validity < THRESHOLDS["citation_validity"]
    or false_insight_rate > THRESHOLDS["false_insight_rate"]
    or stale_memory_leakage_rate > THRESHOLDS["stale_memory_leakage_rate"]
)
if failed:
    raise SystemExit(1)
PY
