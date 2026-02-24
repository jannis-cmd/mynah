#!/usr/bin/env sh
set -eu

sh compute/scripts/ollama-bootstrap.sh

docker compose up -d --build
docker compose ps
