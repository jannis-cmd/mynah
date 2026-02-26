Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "== wearable ble sync =="
@'
import json
import urllib.request

payload = {
    "name_prefix": "MYNAH-WEARABLE",
    "chunk_size": 200,
}
req = urllib.request.Request(
    "http://mynah_agent:8002/sync/wearable_ble",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=90) as resp:
    result = json.loads(resp.read().decode("utf-8"))
print(json.dumps(result, indent=2))
'@ | docker compose exec -T mynah_ui python -
if ($LASTEXITCODE -ne 0) { throw "wearable ble sync failed" }
