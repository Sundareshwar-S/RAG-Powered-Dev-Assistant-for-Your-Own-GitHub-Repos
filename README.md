# CodeBase Oracle

RAG-powered dev assistant for your own GitHub repos. See plan.md for the build plan.

## Local Ollama models

Models are stored in this repo under `ollama_models/` (not `~/.ollama`).

```bash
chmod +x scripts/start-ollama.sh
./scripts/start-ollama.sh
```

To pull a model into the project folder:

```bash
export OLLAMA_MODELS="$(pwd)/ollama_models/models"
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama pull llama3:8b
```
