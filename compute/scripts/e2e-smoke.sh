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

echo "== chunked ingest resume/idempotency + restart durability =="
docker compose exec -T mynah_ui python - <<'PY'
import base64
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
obj1 = "e2e_chunk_note_01"
obj2 = "e2e_chunk_note_02"

with sqlite3.connect("/home/appuser/data/db/mynah.db") as conn:
    conn.execute("DELETE FROM sync_chunk WHERE object_id IN (?, ?)", (obj1, obj2))
    conn.execute("DELETE FROM sync_object WHERE object_id IN (?, ?)", (obj1, obj2))
    conn.execute("DELETE FROM audio_note WHERE id IN (?, ?)", (obj1, obj2))
    conn.commit()

chunk0 = {
    "object_id": obj1,
    "device_id": "fixture_wearable_01",
    "session_id": "e2e_sess_01",
    "start_ts": f"{day}T09:00:00+00:00",
    "end_ts": f"{day}T09:00:08+00:00",
    "chunk_index": 0,
    "total_chunks": 2,
    "chunk_b64": base64.b64encode(b"RIFF_CHUNK_A").decode("ascii"),
    "transcript_hint": "Chunked transfer resumed cleanly.",
    "source": "e2e_smoke",
}
chunk1 = {
    "object_id": obj1,
    "device_id": "fixture_wearable_01",
    "session_id": "e2e_sess_01",
    "start_ts": f"{day}T09:00:00+00:00",
    "end_ts": f"{day}T09:00:08+00:00",
    "chunk_index": 1,
    "total_chunks": 2,
    "chunk_b64": base64.b64encode(b"_B").decode("ascii"),
    "source": "e2e_smoke",
}

def post_chunk(payload):
    req = urllib.request.Request(
        "http://mynahd:8001/ingest/audio_chunk",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

first = post_chunk(chunk0)
assert first["complete"] is False and first["received_chunks"] == 1, first
duplicate = post_chunk(chunk0)
assert duplicate["received_chunks"] == 1, duplicate
with urllib.request.urlopen(f"http://mynahd:8001/ingest/audio_chunk/status?object_id={obj1}", timeout=5) as resp:
    status_mid = json.loads(resp.read().decode("utf-8"))
assert status_mid["sync_object"]["status"] == "pending", status_mid
done = post_chunk(chunk1)
assert done["complete"] is True and done["audio_id"] == obj1, done
print("chunk resume complete", done)
PY

docker compose restart mynahd >/dev/null
sleep 3

docker compose exec -T mynah_ui python - <<'PY'
import base64
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
obj2 = "e2e_chunk_note_02"
chunk0 = {
    "object_id": obj2,
    "device_id": "fixture_wearable_01",
    "session_id": "e2e_sess_02",
    "start_ts": f"{day}T09:10:00+00:00",
    "end_ts": f"{day}T09:10:08+00:00",
    "chunk_index": 0,
    "total_chunks": 2,
    "chunk_b64": base64.b64encode(b"RIFF_RESTART_A").decode("ascii"),
    "transcript_hint": "Chunked transfer survived daemon restart.",
    "source": "e2e_smoke",
}
chunk1 = {
    "object_id": obj2,
    "device_id": "fixture_wearable_01",
    "session_id": "e2e_sess_02",
    "start_ts": f"{day}T09:10:00+00:00",
    "end_ts": f"{day}T09:10:08+00:00",
    "chunk_index": 1,
    "total_chunks": 2,
    "chunk_b64": base64.b64encode(b"_B").decode("ascii"),
    "source": "e2e_smoke",
}

def post_chunk(payload):
    req = urllib.request.Request(
        "http://mynahd:8001/ingest/audio_chunk",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

first = post_chunk(chunk0)
assert first["complete"] is False and first["received_chunks"] == 1, first
print("chunk before restart", first)
PY

docker compose restart mynahd >/dev/null
sleep 3

docker compose exec -T mynah_ui python - <<'PY'
import base64
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
obj2 = "e2e_chunk_note_02"
chunk1 = {
    "object_id": obj2,
    "device_id": "fixture_wearable_01",
    "session_id": "e2e_sess_02",
    "start_ts": f"{day}T09:10:00+00:00",
    "end_ts": f"{day}T09:10:08+00:00",
    "chunk_index": 1,
    "total_chunks": 2,
    "chunk_b64": base64.b64encode(b"_B").decode("ascii"),
    "source": "e2e_smoke",
}
req = urllib.request.Request(
    "http://mynahd:8001/ingest/audio_chunk",
    data=json.dumps(chunk1).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    completed = json.loads(resp.read().decode("utf-8"))
assert completed["complete"] is True and completed["audio_id"] == obj2, completed
with urllib.request.urlopen(f"http://mynahd:8001/ingest/audio_chunk/status?object_id={obj2}", timeout=5) as resp:
    state = json.loads(resp.read().decode("utf-8"))
assert state["sync_object"]["status"] == "complete", state
print("chunk restart recovery complete", completed)
PY

echo "== ingest fixture audio and transcribe =="
docker compose exec -T mynah_ui python - <<'PY'
import base64
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
audio_payload = {
    "note_id": "e2e_note_01",
    "device_id": "fixture_wearable_01",
    "start_ts": f"{day}T10:00:00+00:00",
    "end_ts": f"{day}T10:00:08+00:00",
    "audio_b64": base64.b64encode(b"RIFF_E2E_NOTE").decode("ascii"),
    "transcript_hint": "I walked outside today and felt better afterwards.",
    "source": "e2e_smoke",
}
ingest_req = urllib.request.Request(
    "http://mynahd:8001/ingest/audio",
    data=json.dumps(audio_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(ingest_req, timeout=10) as resp:
    ingest = json.loads(resp.read().decode("utf-8"))
assert ingest["audio_id"] == "e2e_note_01", ingest
print("audio ingest", ingest)

transcribe_req = urllib.request.Request(
    "http://mynah_agent:8002/pipeline/audio/transcribe",
    data=json.dumps({"audio_id": "e2e_note_01", "caller": "e2e_smoke", "force": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(transcribe_req, timeout=20) as resp:
    transcribed = json.loads(resp.read().decode("utf-8"))
assert transcribed["audio_id"] == "e2e_note_01", transcribed
print("audio transcribe", transcribed)

with urllib.request.urlopen("http://mynah_agent:8002/tools/transcript/recent?limit=10", timeout=5) as resp:
    transcripts = json.loads(resp.read().decode("utf-8"))
assert any(item["audio_id"] == "e2e_note_01" for item in transcripts["entries"]), transcripts
print("transcript recent", transcripts["entries"][:2])

memory_search_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_search",
    data=json.dumps({"query": "walked outside", "limit": 10, "verified_only": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(memory_search_req, timeout=10) as resp:
    memory_search = json.loads(resp.read().decode("utf-8"))
assert any(entry["type"] == "note" for entry in memory_search["entries"]), memory_search
print("memory search note", memory_search["entries"][:2])

with urllib.request.urlopen("http://mynah_ui:8000/status", timeout=5) as resp:
    ui_status = json.loads(resp.read().decode("utf-8"))
audio_recent = ui_status.get("audio_recent", {})
entries = audio_recent.get("entries", [])
assert any(entry["id"] == "e2e_note_01" and int(entry["transcript_ready"]) == 1 for entry in entries), ui_status
print("ui audio status", entries[:2])
PY

echo "== report generation and visibility =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.request
from datetime import datetime, timezone

day = datetime.now(timezone.utc).date().isoformat()
generate_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/report_generate",
    data=json.dumps({"date": day, "caller": "e2e_smoke"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(generate_req, timeout=15) as resp:
    report = json.loads(resp.read().decode("utf-8"))
assert report["report_date"] == day, report
assert report["hr_samples"] >= 1, report
print("report generated", report)

with urllib.request.urlopen("http://mynah_agent:8002/tools/report_recent?limit=10", timeout=5) as resp:
    recent = json.loads(resp.read().decode("utf-8"))
assert any(item["report_date"] == day for item in recent["entries"]), recent
print("report recent", recent["entries"][:2])

with urllib.request.urlopen("http://mynah_ui:8000/status", timeout=5) as resp:
    ui_status = json.loads(resp.read().decode("utf-8"))
reports = ui_status.get("reports_recent", {}).get("entries", [])
assert any(item["report_date"] == day for item in reports), ui_status
print("ui reports status", reports[:2])
PY

echo "== sql tool accept/reject/audit =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import urllib.error
import urllib.request

ok_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/sql_query_readonly",
    data=json.dumps(
        {
            "query": "SELECT COUNT(*) AS sample_count FROM hr_sample WHERE device_id = ? LIMIT 1",
            "params": ["fixture_wearable_01"],
            "caller": "e2e_smoke",
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(ok_req, timeout=10) as resp:
    accepted = json.loads(resp.read().decode("utf-8"))
assert accepted["row_count"] == 1, accepted
assert accepted["rows"][0]["sample_count"] == 3, accepted
print("sql accepted", accepted)

for query in ("DELETE FROM hr_sample WHERE 1=1 LIMIT 1", "SELECT * FROM hr_sample"):
    bad_req = urllib.request.Request(
        "http://mynah_agent:8002/tools/sql_query_readonly",
        data=json.dumps({"query": query, "caller": "e2e_smoke"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(bad_req, timeout=10)
        raise AssertionError(f"query should be rejected: {query}")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400, exc.code
        error_payload = json.loads(exc.read().decode("utf-8"))
        print("sql rejected", error_payload["detail"]["code"])

with urllib.request.urlopen("http://mynah_agent:8002/tools/query_audit/recent?limit=20", timeout=5) as resp:
    audit = json.loads(resp.read().decode("utf-8"))
entries = [entry for entry in audit["entries"] if entry["caller"] == "e2e_smoke"]
assert any(entry["status"] == "accepted" for entry in entries), audit
assert sum(1 for entry in entries if entry["status"] == "rejected") >= 2, audit
print("query audit entries", entries[:3])
PY

echo "== memory tool governance/verify/supersession =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import time
import urllib.error
import urllib.request

run_id = int(time.time())

invalid_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_upsert",
    data=json.dumps(
        {
            "type": "insight",
            "title": "Insufficient citations",
            "summary": "This should fail because it has only one citation.",
            "tags": ["hr"],
            "sensitivity": "personal",
            "salience_score": 0.8,
            "confidence_score": 0.9,
            "caller": "e2e_smoke",
            "citations": [
                {
                    "source_type": "hr_sample",
                    "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                    "content_hash": "11111111aaaaaaaa",
                    "schema_version": 1,
                    "snapshot_ref": "2026-02-24",
                }
            ],
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(invalid_req, timeout=10)
    raise AssertionError("insight upsert should have been rejected")
except urllib.error.HTTPError as exc:
    assert exc.code == 400, exc.code
    payload = json.loads(exc.read().decode("utf-8"))
    assert payload["detail"]["code"] == "MEMORY_CITATION_MIN_NOT_MET", payload
    print("memory rejected", payload["detail"]["code"])

first_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_upsert",
    data=json.dumps(
        {
            "type": "fact",
            "title": f"Morning resting HR {run_id}",
            "summary": f"Typical resting HR is low 60s. run={run_id}",
            "tags": ["hr", "resting"],
            "sensitivity": "personal",
            "salience_score": 0.8,
            "confidence_score": 0.9,
            "caller": "e2e_smoke",
            "citations": [
                {
                    "source_type": "hr_sample",
                    "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                    "content_hash": "22222222bbbbbbbb",
                    "schema_version": 1,
                    "snapshot_ref": "2026-02-24",
                }
            ],
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(first_req, timeout=10) as resp:
    first = json.loads(resp.read().decode("utf-8"))
first_id = first["memory_id"]
print("memory accepted first", first)

second_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_upsert",
    data=json.dumps(
        {
            "type": "fact",
            "title": f"Morning resting HR {run_id}",
            "summary": f"Updated resting HR baseline is mid 60s. run={run_id}",
            "tags": ["hr", "resting"],
            "sensitivity": "personal",
            "salience_score": 0.82,
            "confidence_score": 0.93,
            "caller": "e2e_smoke",
            "supersedes_memory_id": first_id,
            "citations": [
                {
                    "source_type": "hr_sample",
                    "source_ref": "fixture_wearable_01|2026-02-24T00:00:02+00:00",
                    "content_hash": "33333333cccccccc",
                    "schema_version": 1,
                    "snapshot_ref": "2026-02-24",
                }
            ],
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(second_req, timeout=10) as resp:
    second = json.loads(resp.read().decode("utf-8"))
second_id = second["memory_id"]
print("memory accepted second", second)

with urllib.request.urlopen(f"http://mynah_agent:8002/tools/memory_verify/{first_id}", timeout=5) as resp:
    first_verify = json.loads(resp.read().decode("utf-8"))
with urllib.request.urlopen(f"http://mynah_agent:8002/tools/memory_verify/{second_id}", timeout=5) as resp:
    second_verify = json.loads(resp.read().decode("utf-8"))
assert first_verify["active"] is False and first_verify["superseded_by"] == second_id, first_verify
assert second_verify["active"] is True and second_verify["verified"] is True, second_verify
print("memory verify", {"first": first_verify, "second": second_verify})

search_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_search",
    data=json.dumps({"query": "resting", "limit": 10, "verified_only": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(search_req, timeout=10) as resp:
    search = json.loads(resp.read().decode("utf-8"))
assert len(search["entries"]) >= 1, search
assert any(entry["id"] == second_id for entry in search["entries"]), search
assert all(entry["id"] != first_id for entry in search["entries"]), search
print("memory search", search["entries"][:2])
PY

echo "== stale memory exclusion and reverification =="
docker compose exec -T mynah_ui python - <<'PY'
import json
import sqlite3
import time
import urllib.request

run_id = int(time.time())
create_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_upsert",
    data=json.dumps(
        {
            "type": "fact",
            "title": f"Freshness baseline {run_id}",
            "summary": f"Freshness test memory run={run_id}",
            "tags": ["freshness", "hr"],
            "sensitivity": "personal",
            "salience_score": 0.8,
            "confidence_score": 0.9,
            "caller": "e2e_smoke",
            "citations": [
                {
                    "source_type": "hr_sample",
                    "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                    "content_hash": "freshnessaaaa1111",
                    "schema_version": 1,
                    "snapshot_ref": "2026-02-24",
                }
            ],
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(create_req, timeout=10) as resp:
    created = json.loads(resp.read().decode("utf-8"))
memory_id = created["memory_id"]
print("freshness memory created", created)

with sqlite3.connect("/home/appuser/data/db/mynah.db") as conn:
    conn.execute(
        "UPDATE memory_item SET updated_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00+00:00", memory_id),
    )
    conn.commit()

with urllib.request.urlopen(f"http://mynah_agent:8002/tools/memory_verify/{memory_id}", timeout=5) as resp:
    stale_verify = json.loads(resp.read().decode("utf-8"))
assert stale_verify["stale"] is True and stale_verify["verified"] is False, stale_verify
print("stale verify", stale_verify)

search_stale_req = urllib.request.Request(
    "http://mynah_agent:8002/tools/memory_search",
    data=json.dumps({"query": str(run_id), "limit": 10, "verified_only": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(search_stale_req, timeout=10) as resp:
    search_stale = json.loads(resp.read().decode("utf-8"))
assert all(entry["id"] != memory_id for entry in search_stale["entries"]), search_stale
print("stale search excluded", search_stale["entries"][:2])

reverify_req = urllib.request.Request(
    f"http://mynah_agent:8002/tools/memory_reverify/{memory_id}",
    data=b"",
    method="POST",
)
with urllib.request.urlopen(reverify_req, timeout=10) as resp:
    reverified = json.loads(resp.read().decode("utf-8"))
assert reverified["verification"]["verified"] is True, reverified
assert reverified["verification"]["stale"] is False, reverified
print("reverified", reverified)

with urllib.request.urlopen(f"http://mynah_agent:8002/tools/memory_verify/{memory_id}", timeout=5) as resp:
    fresh_verify = json.loads(resp.read().decode("utf-8"))
assert fresh_verify["verified"] is True and fresh_verify["stale"] is False, fresh_verify
print("fresh verify", fresh_verify)
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
