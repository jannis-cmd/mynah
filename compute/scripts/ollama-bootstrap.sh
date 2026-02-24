#!/usr/bin/env sh
set -eu

MODEL="qwen2.5:7b"
VOLUME="mynah_ollama_models"

docker volume create "$VOLUME" >/dev/null
docker run --rm -v "$VOLUME:/root/.ollama" --entrypoint /bin/sh ollama/ollama:latest \
  -c "ollama serve >/tmp/ollama.log 2>&1 & sleep 2; ollama pull $MODEL"
