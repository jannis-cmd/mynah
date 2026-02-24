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
    "http://mynah_ui:8000/status",
):
    with urllib.request.urlopen(url, timeout=5) as resp:
        print(url, json.loads(resp.read().decode("utf-8")))
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
