Write-Host "== services =="
docker compose ps

Write-Host "== service health endpoints =="
docker compose exec -T mynah_ui python -c "import json,urllib.request; urls=['http://mynahd:8001/health','http://mynah_agent:8002/health','http://mynah_ui:8000/health','http://mynah_ui:8000/status']; [print(u, json.loads(urllib.request.urlopen(u, timeout=5).read().decode('utf-8'))) for u in urls]"

Write-Host "== agent analyze =="
docker compose exec -T mynah_ui python -c "import json,urllib.request; req=urllib.request.Request('http://mynah_agent:8002/analyze', data=json.dumps({'prompt':'Reply with exactly: E2E_OK'}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(json.loads(urllib.request.urlopen(req, timeout=60).read().decode('utf-8')))"
