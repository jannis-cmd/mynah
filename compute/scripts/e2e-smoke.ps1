Write-Host "== services =="
docker compose ps

Write-Host "== service health endpoints =="
docker compose exec -T mynah_ui python -c "import json,urllib.request; urls=['http://mynahd:8001/health','http://mynah_agent:8002/health','http://mynah_ui:8000/health']; [print(u, json.loads(urllib.request.urlopen(u, timeout=5).read().decode('utf-8'))) for u in urls]"

Write-Host "== ingest fixture HR =="
docker compose exec -T mynah_ui python -c "import json,urllib.request,datetime; d=datetime.datetime.now(datetime.timezone.utc).date().isoformat(); samples=[{'ts':f'{d}T00:00:01+00:00','bpm':62,'quality':95,'sensor_status':'ok'},{'ts':f'{d}T00:00:02+00:00','bpm':67,'quality':95,'sensor_status':'ok'},{'ts':f'{d}T00:00:03+00:00','bpm':71,'quality':95,'sensor_status':'ok'}]; payload={'device_id':'fixture_wearable_01','source':'e2e_smoke','samples':samples}; req=urllib.request.Request('http://mynahd:8001/ingest/hr', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(json.loads(urllib.request.urlopen(req, timeout=10).read().decode('utf-8')))"

Write-Host "== verify HR summary and UI status =="
docker compose exec -T mynah_ui python -c "import json,urllib.request,datetime; d=datetime.datetime.now(datetime.timezone.utc).date().isoformat(); url=f'http://mynahd:8001/summary/hr/today?date={d}&device_id=fixture_wearable_01'; summary=json.loads(urllib.request.urlopen(url, timeout=5).read().decode('utf-8')); assert summary['sample_count']==3, summary; print('daemon summary', summary); ui=json.loads(urllib.request.urlopen('http://mynah_ui:8000/status', timeout=5).read().decode('utf-8')); assert ui.get('hr_today') is not None, ui; print('ui status', ui)"

Write-Host "== agent analyze =="
docker compose exec -T mynah_ui python -c "import json,urllib.request; req=urllib.request.Request('http://mynah_agent:8002/analyze', data=json.dumps({'prompt':'Reply with exactly: E2E_OK'}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST'); print(json.loads(urllib.request.urlopen(req, timeout=60).read().decode('utf-8')))"
