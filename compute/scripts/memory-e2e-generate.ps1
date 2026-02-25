Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

python tools/test-harness/memory_e2e/generate_testsets.py `
  --output-root storage/test_data/memory_e2e `
  --sessions-root "$env:USERPROFILE\.codex\sessions" `
  --ollama-container mynah-ollama-1 `
  --model qwen2.5:7b `
  --count 200 `
  --start-date 2025-08-01 `
  --timezone UTC `
  --seed 20260225 `
  --batch-size 10

if ($LASTEXITCODE -ne 0) {
  throw "dataset generation failed"
}
