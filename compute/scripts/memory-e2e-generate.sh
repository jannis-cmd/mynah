#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python tools/test-harness/memory_e2e/generate_testsets.py \
  --output-root storage/test_data/memory_e2e \
  --sessions-root "$HOME/.codex/sessions" \
  --ollama-container mynah-ollama-1 \
  --model qwen2.5:7b \
  --count 200 \
  --start-date 2025-08-01 \
  --timezone UTC \
  --seed 20260225 \
  --batch-size 10
