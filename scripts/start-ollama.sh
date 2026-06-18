#!/usr/bin/env bash
# Start Ollama with models stored in this project's ollama_models/ directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_MODELS="${ROOT}/ollama_models/models"
mkdir -p "${OLLAMA_MODELS}"

if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama already running at http://localhost:11434"
  echo "Models dir: ${OLLAMA_MODELS}"
  exit 0
fi

echo "Starting Ollama (models: ${OLLAMA_MODELS})"
ollama serve > "${ROOT}/ollama_models/ollama.log" 2>&1 &

# Poll up to 30 seconds (10 attempts × 3 s) instead of a fixed sleep.
RETRIES=10
for i in $(seq 1 $RETRIES); do
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama ready at http://localhost:11434 (attempt ${i})"
    ollama list
    exit 0
  fi
  echo "Waiting for Ollama… (${i}/${RETRIES})"
  sleep 3
done

echo "Ollama did not become ready after $((RETRIES * 3))s. See ${ROOT}/ollama_models/ollama.log"
exit 1
