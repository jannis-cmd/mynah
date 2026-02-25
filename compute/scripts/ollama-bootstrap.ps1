$GenModel = "qwen2.5:7b"
$EmbedModel = "qwen3-embedding:0.6b"
$Volume = "mynah_ollama_models"

docker volume create $Volume | Out-Null
docker run --rm -v "${Volume}:/root/.ollama" --entrypoint /bin/sh ollama/ollama:latest -c "ollama serve >/tmp/ollama.log 2>&1 & sleep 2; ollama pull $GenModel && ollama pull $EmbedModel"
