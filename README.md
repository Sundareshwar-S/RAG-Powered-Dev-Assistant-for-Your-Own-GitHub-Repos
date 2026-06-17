# CodeBase Oracle

CodeBase Oracle is a local RAG-powered dev assistant for your own GitHub repositories. It clones a repo, splits source files into AST-aware code chunks with tree-sitter, embeds them locally with sentence-transformers (`nomic-ai/nomic-embed-text-v1`, 768-dim), and stores vectors in ChromaDB. When you ask a question, a hybrid retriever combines BM25 keyword search, dense vector search, reciprocal rank fusion (RRF), and cross-encoder reranking to find the most relevant code, then streams an answer from a local LLM via Ollama (default: Qwen2.5-Coder 7B). A React UI handles repo indexing, SSE ingestion progress, WebSocket chat, and source code cards with file paths and line numbers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose (5 services)                        │
├──────────────┬──────────────┬──────────────┬──────────────┬───────────────┤
│   frontend   │   backend    │    chroma    │    ollama    │ model-puller  │
│  React+Vite  │   FastAPI    │  ChromaDB    │  LLM+embed   │  one-shot     │
│   :5173      │   :8000      │  :8001→8000  │   :11434     │  model pull   │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴───────────────┘
       │              │              │              │
       │  REST/WS     │              │              │
       └──────────────►              │              │
                      │              │              │
         INGEST       │              │              │
         ─────────────►              │              │
                      │  clone repo  │              │
                      │  AST chunk   │              │
                      │  embed (local)              │
                      │  upsert ────►│ vectors      │
                      │  BM25 cache  │              │
                      │  (bm25_cache/)             │
                      │              │              │
         QUERY        │              │              │
         ─────────────►              │              │
                      │  hybrid retrieve           │
                      │  (BM25 + dense + RRF       │
                      │   + cross-encoder)         │
                      │  prompt + stream ─────────►│ qwen2.5-coder:7b
                      │              │              │ llama3:8b
       ◄──────────────┤  answer + sources          │
```

**Persistent volumes**

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./chroma_data` | `/chroma_db` | ChromaDB vector store |
| `./bm25_cache` | `/bm25_cache` | BM25 index JSON (`repo_{id}.json`) |
| `./repos_cache` | `/tmp/repos` | Ephemeral git clones |
| `./ollama_models` | `/root/.ollama` | Ollama model blobs |

---

## Prerequisites

- **Docker** and **Docker Compose** (recommended path)
- **~20 GB disk** — models ~9 GB after pull, plus Chroma/BM25 data
- **GPU optional** — default `qwen2.5-coder:7b` runs on CPU / low-VRAM hosts
- **Optional:** `GITHUB_TOKEN` for private repos and GitHub API rate limits

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

1. Wait for all healthchecks to pass. On first boot, the `model-puller` service downloads three models (10–30+ minutes depending on bandwidth).
2. Confirm models loaded inside Compose (after `model-puller` exits):

```bash
docker compose exec ollama ollama list
```

If you also run host Ollama on port 11434, the Compose stack keeps its Ollama internal-only to avoid port conflicts.
3. Open the UI: [http://localhost:5173](http://localhost:5173)
4. Confirm backend health:

```bash
curl -s http://localhost:8000/api/v1/health
# {"status":"ok","ollama":"ok","chroma":"ok"}
```

### Host-only Ollama (optional dev path)

Models live in this repo under `ollama_models/` (not `~/.ollama`):

```bash
chmod +x scripts/start-ollama.sh
./scripts/start-ollama.sh

export OLLAMA_MODELS="$(pwd)/ollama_models/models"
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama pull llama3:8b
```

When running the backend on the host (outside Docker), set `OLLAMA_URL=http://localhost:11434` in `.env`.

---

## First Ingest

Index a public repository via the UI (sidebar → paste URL → **Index Repo**) or the API:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/pallets/markupsafe","branch":"main"}'
```

Response (HTTP 202):

```json
{"job_id": "<uuid>", "status": "running"}
```

Stream ingestion progress (SSE):

```bash
curl -N http://localhost:8000/api/v1/ingest/<job_id>/status
```

The `repo_id` for markupsafe is the first 8 characters of `md5(repo_url)` → **`24f35f55`**.

If the same repo is already ingesting, the API returns **409 Conflict** with the existing `job_id`.

---

## First Query

Non-streaming query:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "24f35f55",
    "question": "How does escape work?",
    "model": "qwen2.5-coder:7b"
  }'
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "file_path": "src/markupsafe/__init__.py",
      "start_line": 10,
      "end_line": 45,
      "chunk_type": "function_definition",
      "symbol_name": "escape",
      "text": "..."
    }
  ],
  "model_used": "qwen2.5-coder:7b"
}
```

**WebSocket streaming chat** — connect to `ws://localhost:8000/api/v1/ws/chat`, send:

```json
{"repo_id": "24f35f55", "question": "How does escape work?", "model": "qwen2.5-coder:7b"}
```

Message sequence: `{"type":"sources","sources":[...]}` → `{"type":"token","token":"..."}` (repeated) → `{"type":"done"}`.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/ingest` | Start background ingestion (409 if already running) |
| GET | `/api/v1/ingest/{job_id}/status` | SSE progress stream |
| POST | `/api/v1/query` | Non-streaming RAG query |
| WS | `/api/v1/ws/chat` | Streaming chat |
| GET | `/api/v1/repos` | List indexed repos |
| DELETE | `/api/v1/repos/{repo_id}` | Delete collection + BM25 cache |
| GET | `/api/v1/health` | Ollama + ChromaDB connectivity |

Rate limits: **5 ingest requests/minute**, **20 query requests/minute** (per client IP).

---

## Environment Variables

Copy [`.env.example`](.env.example) to `.env`. Never commit `.env`.

| Variable | Default (Docker) | Purpose |
|----------|------------------|---------|
| `GITHUB_TOKEN` | *(empty)* | Private repo clone; GitHub size check |
| `OLLAMA_URL` | `http://ollama:11434` | LLM chat inference (embeddings run in-process in the backend) |
| `CHROMA_HOST` | `http://chroma:8000` | ChromaDB HTTP API (internal hostname) |
| `DEFAULT_LLM_MODEL` | `qwen2.5-coder:7b` | Default chat model |
| `EMBED_LOCAL_MODEL` | `nomic-ai/nomic-embed-text-v1` | Local sentence-transformers embed model (768-dim) |
| `EMBED_LOCAL_BATCH_SIZE` | `64` | In-process embed batch size during ingest/query |
| `EMBED_LOCAL_DEVICE` | `cpu` | Device for local embeddings (`cpu` or `cuda`) |
| `MAX_REPO_SIZE_MB` | `500` | Reject repos larger than this |
| `BM25_CACHE_DIR` | `/bm25_cache` | Persistent BM25 JSON cache directory |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `OLLAMA_MODELS` | `./ollama_models/models` | Host-only: project-local model storage |

**Host overrides:** When running backend/Ollama outside Docker, use `OLLAMA_URL=http://localhost:11434`. ChromaDB is exposed on the host at `http://localhost:8001`.

---

## Hardware Requirements

| Model | Approx. size | Role |
|-------|--------------|------|
| `qwen2.5-coder:7b` | ~6 GB | Default code LLM (CPU/low-VRAM friendly) |
| `qwen2.5-coder:14b` | ~11 GB Q4 | Optional high-VRAM alternative |
| `llama3:8b` | ~4.7 GB | Alternative chat model (UI selector) |
| `nomic-embed-text` | ~0.3 GB | Legacy Ollama embed fallback (primary path is local `nomic-ai/nomic-embed-text-v1`) |

**Ingest tuning:** Large repos (10k+ chunks) embed in-process in the backend — no per-chunk Ollama HTTP. Increase `EMBED_LOCAL_BATCH_SIZE` (e.g. `128`) on machines with more RAM for faster indexing. SSE progress reports three phases: chunking (0–20%), embedding (20–90%), BM25 build (90–100%).

---

## Evaluation

Golden QA test sets live in [`tests/golden_qa/`](tests/golden_qa/). See [`tests/golden_qa/README.md`](tests/golden_qa/README.md) for schema and how to add new repos.

After indexing markupsafe (`repo_id=24f35f55`):

```bash
# Retrieval quality (requires running stack + indexed repo)
python tests/eval_retrieval.py

# Generation quality
python tests/eval_generation.py

# Offline integration tests (no running stack)
pytest tests/test_phase1_integration.py tests/test_phase2_retrieval.py -v
```

**Targets:** Recall@5 > 0.70, MRR > 0.60, keyword_hit_rate > 0.80.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Models not ready / embed fails | Wait for `model-puller` to finish: `docker compose logs model-puller`; then `docker compose exec ollama ollama list` |
| Port 11434 already in use | Host Ollama and Compose Ollama cannot both bind `:11434`; Compose uses internal-only Ollama — stop host `ollama serve` only if you need the published port |
| `KeyError: 'dimension'` on query | ChromaDB client and Docker image version mismatch — both must be **0.5.23**; do not set deprecated `CHROMA_DB_IMPL` on the Chroma service |
| Slow first query after restart | BM25 should load from `./bm25_cache/repo_*.json`; verify volume mount in `docker-compose.yml` |
| Ingest returns 422 | Only **HTTPS** URLs on `github.com`, `gitlab.com`, or `bitbucket.org` |
| Repo too large | Default limit 500 MB (`MAX_REPO_SIZE_MB`); use a smaller repo or raise the limit |
| HTTP 429 | Rate limit exceeded — ingest 5/min, query 20/min |
| Healthcheck never passes | Check `docker compose ps`; Ollama/Chroma healthchecks use `curl` inside containers |
| Empty retrieval / low-confidence answer | Guardrail returns a canned message when best chunk score < 0.10 |

---

## Research & References

- **BM25 / Okapi BM25** — Robertson, S. & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. *SIGIR*.
- **Reciprocal Rank Fusion** — Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*.
- **Cross-encoder reranking** — Reimers, N. & Gurevych, I. — `cross-encoder/ms-marco-MiniLM-L-6-v2` via [sentence-transformers](https://www.sbert.net/).
- **AST chunking** — [tree-sitter](https://tree-sitter.github.io/) grammars for Python, JS/TS, Java, Go, Rust.
- **Embeddings** — [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama (768 dimensions).
- **Local LLM inference** — [Ollama](https://ollama.com/) with Qwen2.5-Coder and Llama 3.

---

## Verification

*Updated during Phase 9 completion check — see checklist results below.*

| Check | Status |
|-------|--------|
| `docker compose up --build` | *pending* |
| Ollama models (3) | *pending* |
| UI ingest + SSE progress | *pending* |
| Duplicate ingest 409 | *pending* |
| WebSocket streaming + source cards | *pending* |
| Model switch (Qwen / Llama3) | *pending* |
| Delete repo | *pending* |
| Health endpoint | *pending* |
| BM25 survives backend restart | *pending* |
| Recall@5 / MRR | *pending* |
| keyword_hit_rate | *pending* |
| No secrets in git | *pending* |
