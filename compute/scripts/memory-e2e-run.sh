#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET_HOST="$REPO_ROOT/storage/test_data/memory_e2e"
CONTAINER_NAME="mynah-mynah_agent-1"
DATASET_CONTAINER="/home/appuser/data/artifacts/test_data/memory_e2e"
REPORT_CONTAINER="/home/appuser/data/artifacts/reports/memory-e2e-report.md"
REPORT_HOST_DIR="$REPO_ROOT/reports"
REPORT_HOST="$REPORT_HOST_DIR/memory-e2e-report.md"

if [ ! -d "$DATASET_HOST" ]; then
  echo "dataset root not found: $DATASET_HOST" >&2
  exit 1
fi

OLLAMA_MODEL="qwen2.5:7b" docker compose up -d --force-recreate mynah_agent

docker exec --user root "$CONTAINER_NAME" sh -lc "rm -rf $DATASET_CONTAINER && mkdir -p /home/appuser/data/artifacts/test_data /home/appuser/data/artifacts/reports"
docker cp "$DATASET_HOST" "$CONTAINER_NAME:/home/appuser/data/artifacts/test_data"

cat tools/test-harness/memory_e2e/run_ingest_and_report.py | docker exec -i "$CONTAINER_NAME" python - \
  --dataset-root "$DATASET_CONTAINER" \
  --api-base-url "http://127.0.0.1:8002" \
  --report-path "$REPORT_CONTAINER" \
  --max-codex-artifacts 20 \
  --reset-db

mkdir -p "$REPORT_HOST_DIR"
docker cp "$CONTAINER_NAME:$REPORT_CONTAINER" "$REPORT_HOST"
echo "Report written to $REPORT_HOST"
