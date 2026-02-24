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
import urllib.error
import urllib.request

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
            "title": "Morning resting HR",
            "summary": "Typical resting HR is low 60s.",
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
            "title": "Morning resting HR",
            "summary": "Updated resting HR baseline is mid 60s.",
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
