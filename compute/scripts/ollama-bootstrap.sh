#!/usr/bin/env sh
set -eu

GEN_MODEL="qwen2.5:7b"
EMBED_MODEL="qwen3-embedding:0.6b"
VOLUME="mynah_ollama_models"

docker volume create "$VOLUME" >/dev/null
docker run --rm -v "$VOLUME:/root/.ollama" --entrypoint /bin/sh ollama/ollama:latest \
  -c "ollama serve >/tmp/ollama.log 2>&1 & sleep 2; ollama pull $GEN_MODEL && ollama pull $EMBED_MODEL"
