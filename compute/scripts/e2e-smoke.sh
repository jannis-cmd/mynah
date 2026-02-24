#!/usr/bin/env sh
set -eu

echo "== services =="
docker compose ps

echo "== daemon status =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.request

for url in (
    "http://mynahd:8001/health",
    "http://mynah_agent:8002/health",
    "http://mynah_ui:8000/health",
):
    with urllib.request.urlopen(url, timeout=5) as resp:
        print(url, json.loads(resp.read().decode("utf-8")))
PY

echo "== ingest fixture HR =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
payload = {
    "device_id": "fixture_wearable_01",
    "source": "e2e_smoke",
    "samples": [
        {"ts": f"{day}T00:00:01+00:00", "bpm": 62, "quality": 95, "sensor_status": "ok"},
        {"ts": f"{day}T00:00:02+00:00", "bpm": 67, "quality": 95, "sensor_status": "ok"},
        {"ts": f"{day}T00:00:03+00:00", "bpm": 71, "quality": 95, "sensor_status": "ok"},
    ],
}
req = urllib.request.Request(
    "http://mynahd:8001/ingest/hr",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    print(json.loads(resp.read().decode("utf-8")))
PY

echo "== verify HR summary and UI status =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
summary_url = f"http://mynahd:8001/summary/hr/today?date={day}&device_id=fixture_wearable_01"
with urllib.request.urlopen(summary_url, timeout=5) as resp:
    summary = json.loads(resp.read().decode("utf-8"))
assert summary["sample_count"] == 3, summary
print("daemon summary", summary)

with urllib.request.urlopen("http://mynah_ui:8000/status", timeout=5) as resp:
    ui_status = json.loads(resp.read().decode("utf-8"))
assert ui_status.get("hr_today") is not None, ui_status
print("ui status", ui_status)
PY

echo "== agent analyze =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.request

req = urllib.request.Request(
    "http://mynah_agent:8002/analyze",
    data=json.dumps({"prompt": "Reply with exactly: E2E_OK"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
print(payload)
PY
