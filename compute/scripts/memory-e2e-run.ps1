Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$datasetHost = Resolve-Path "storage/test_data/memory_e2e"
$containerName = "mynah-mynah_agent-1"
$datasetContainer = "/home/appuser/data/artifacts/test_data/memory_e2e"
$reportContainer = "/home/appuser/data/artifacts/reports/memory-e2e-report.md"
$reportHostDir = Join-Path $repoRoot "reports"
$reportHost = Join-Path $reportHostDir "memory-e2e-report.md"

if (-not (Test-Path $datasetHost)) {
  throw "dataset root not found: $datasetHost"
}

# Model override for E2E stability on this runtime.
$env:OLLAMA_MODEL = "qwen2.5:7b"
docker compose up -d --force-recreate mynah_agent
if ($LASTEXITCODE -ne 0) {
  throw "failed to recreate mynah_agent with OLLAMA_MODEL override"
}
Remove-Item Env:OLLAMA_MODEL -ErrorAction SilentlyContinue

docker exec $containerName sh -lc "mkdir -p $datasetContainer /home/appuser/data/artifacts/reports"
docker exec --user root $containerName sh -lc "rm -rf $datasetContainer && mkdir -p /home/appuser/data/artifacts/test_data /home/appuser/data/artifacts/reports"
docker cp "$datasetHost" "${containerName}:/home/appuser/data/artifacts/test_data"

Get-Content -Raw "tools/test-harness/memory_e2e/run_ingest_and_report.py" | docker exec -i $containerName python - `
  --dataset-root "$datasetContainer" `
  --api-base-url "http://127.0.0.1:8002" `
  --report-path "$reportContainer" `
  --max-codex-artifacts 20 `
  --reset-db

if ($LASTEXITCODE -ne 0) {
  throw "E2E ingest/report run failed"
}

New-Item -ItemType Directory -Force -Path $reportHostDir | Out-Null
docker cp "${containerName}:$reportContainer" "$reportHost"
Write-Host "Report written to $reportHost"
