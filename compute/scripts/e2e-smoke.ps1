Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "== services =="
docker compose ps

Write-Host "== health =="
@'
import json
import urllib.request

for url in (
    "http://mynah_agent:8002/health",
    "http://mynah_agent:8002/ready",
    "http://mynah_ui:8000/health",
):
    with urllib.request.urlopen(url, timeout=10) as resp:
        print(url, json.loads(resp.read().decode("utf-8")))
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "health check block failed" }

Write-Host "== hr ingest and summary =="
@'
import json
import time
import urllib.request
from datetime import datetime, timezone
from datetime import datetime, timezone

run_id = int(time.time())
device_id = f"fixture_wearable_{run_id}"
day = datetime.now(timezone.utc).date().isoformat()

payload = {
    "device_id": device_id,
    "source": "e2e_smoke",
    "samples": [
        {"ts": f"{day}T00:00:01+00:00", "bpm": 62, "quality": 95, "sensor_status": "ok"},
        {"ts": f"{day}T00:00:02+00:00", "bpm": 67, "quality": 95, "sensor_status": "ok"},
        {"ts": f"{day}T00:00:03+00:00", "bpm": 71, "quality": 95, "sensor_status": "ok"},
    ],
}
req = urllib.request.Request(
    "http://mynah_agent:8002/ingest/hr",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=15) as resp:
    ingest = json.loads(resp.read().decode("utf-8"))
assert ingest["inserted"] == 3, ingest
print("ingest", ingest)

with urllib.request.urlopen(f"http://mynah_agent:8002/summary/hr/today?date={day}&device_id={device_id}", timeout=10) as resp:
    summary = json.loads(resp.read().decode("utf-8"))
assert summary["sample_count"] == 3, summary
print("summary", summary)
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "hr ingest block failed" }

Write-Host "== audio -> transcript -> write plan =="
@'
import base64
import json
import time
import urllib.request
from datetime import datetime, timezone

run_id = int(time.time())
note_id = f"e2e_note_{run_id}"
day = datetime.now(timezone.utc).date().isoformat()

audio_payload = {
    "note_id": note_id,
    "device_id": "fixture_wearable_01",
    "start_ts": f"{day}T10:00:00+00:00",
    "end_ts": f"{day}T10:00:08+00:00",
    "audio_b64": base64.b64encode(b"RIFF_E2E_NOTE").decode("ascii"),
    "transcript_hint": "Today I slept better after reducing evening caffeine and walking after lunch.",
    "source": "e2e_smoke",
}
ingest_req = urllib.request.Request(
    "http://mynah_agent:8002/ingest/audio",
    data=json.dumps(audio_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(ingest_req, timeout=15) as resp:
    ingest = json.loads(resp.read().decode("utf-8"))
assert ingest["audio_id"] == note_id, ingest
print("audio ingest", ingest)

transcribe_req = urllib.request.Request(
    "http://mynah_agent:8002/pipeline/audio/transcribe",
    data=json.dumps({"audio_id": note_id, "caller": "e2e_smoke", "force": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(transcribe_req, timeout=240) as resp:
    processed = json.loads(resp.read().decode("utf-8"))
assert processed["status"] == "ok", processed
print("audio processed", processed)

with urllib.request.urlopen("http://mynah_agent:8002/tools/transcript/recent?limit=20", timeout=10) as resp:
    recent = json.loads(resp.read().decode("utf-8"))
assert any(item["audio_id"] == note_id for item in recent["entries"]), recent
print("transcript recent", recent["entries"][:2])
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "audio pipeline block failed" }

Write-Host "== me.md artifact process =="
@'
import json
import time
import urllib.request

run_id = int(time.time())
markdown = (
    f"# Daily note {run_id}\n"
    "Mood was calmer in the afternoon.\n"
    "I had less shoulder pain on days with stretching.\n"
)
req = urllib.request.Request(
    "http://mynah_agent:8002/pipeline/me_md/process",
    data=json.dumps({
        "source_type": "manual_text",
        "markdown": markdown,
        "upload_ts": datetime.now(timezone.utc).isoformat(),
        "source_ts": None,
        "day_scope": False,
        "timezone": "UTC",
        "caller": "e2e_smoke"
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    out = json.loads(resp.read().decode("utf-8"))
assert out["status"] == "ok", out
print("me_md", out)
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "me_md block failed" }

Write-Host "== report and ui status =="
@'
import json
import urllib.request
from datetime import datetime, timezone

report_date = datetime.now(timezone.utc).date().isoformat()
req = urllib.request.Request(
    "http://mynah_agent:8002/tools/report_generate",
    data=json.dumps({"date": report_date, "caller": "e2e_smoke"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    report = json.loads(resp.read().decode("utf-8"))
assert report["status"] == "ok", report
print("report", report)

with urllib.request.urlopen("http://mynah_agent:8002/tools/report_recent?limit=5", timeout=10) as resp:
    recent = json.loads(resp.read().decode("utf-8"))
assert any(item["report_date"] == report_date for item in recent["entries"]), recent
print("report recent", recent["entries"][:2])

with urllib.request.urlopen("http://mynah_ui:8000/status", timeout=10) as resp:
    ui_status = json.loads(resp.read().decode("utf-8"))
assert ui_status.get("hr_today") is not None, ui_status
assert ui_status.get("reports_recent") is not None, ui_status
print("ui status", ui_status)
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "report/ui block failed" }
