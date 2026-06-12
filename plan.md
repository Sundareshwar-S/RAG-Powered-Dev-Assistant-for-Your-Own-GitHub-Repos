# CodeBase Oracle — AI Agent Execution Plan

> **What this is:** A step-by-step, copy-paste-ready execution plan for an AI agent to build the CodeBase Oracle project from zero to running.
> **Source:** Based on the original `codebase_oracle_blueprint (1).md` — all architecture, tech choices, and code are taken directly from that document.
> **Rule for the agent:** Execute steps in exact order. Do NOT skip ahead. Complete each step's verification before moving to the next.

---

## Tech Stack (Do Not Change)

| Component | Technology | Version |
|---|---|---|
| Backend | Python + FastAPI | Python 3.11, FastAPI 0.115.0 |
| Vector DB | ChromaDB (persistent, file-based) | chromadb 0.5.0 |
| Embeddings | nomic-embed-text via Ollama | 768-dim, 8192 token context |
| LLM | CodeLlama 13B + Llama3 8B via Ollama | Local inference only |
| Retrieval | Hybrid BM25 + Dense + RRF + Cross-encoder reranker | rank-bm25 0.2.2, sentence-transformers 3.0.0 |
| Frontend | React + Vite | Node 20, Vite latest |
| Containers | Docker Compose | 5 services |
| AST Parsing | tree-sitter | Per-language grammars |

---

## Project File Structure (Create This Exactly)

```
codebase-oracle/
├── docker-compose.yml
├── .env
├── .gitignore
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_ingest.py
│   │   ├── routes_query.py
│   │   └── routes_repos.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── git_cloner.py
│   │   ├── file_walker.py
│   │   ├── ast_chunker.py
│   │   ├── embedding_service.py
│   │   ├── chroma_writer.py
│   │   └── bm25_builder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── rrf_fusion.py
│   │   ├── reranker.py
│   │   ├── prompt_builder.py
│   │   └── ollama_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   └── logger.py
│   └── jobs/
│       ├── __init__.py
│       └── job_store.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   ├── Sidebar/
│       │   │   ├── RepoManager.jsx
│       │   │   ├── IngestionProgress.jsx
│       │   │   └── IndexedRepoList.jsx
│       │   ├── Chat/
│       │   │   ├── ChatWindow.jsx
│       │   │   ├── UserMessage.jsx
│       │   │   ├── AssistantMessage.jsx
│       │   │   ├── StreamingCursor.jsx
│       │   │   └── QueryInput.jsx
│       │   └── Sources/
│       │       ├── SourceCards.jsx
│       │       └── CodeSnippetCard.jsx
│       ├── hooks/
│       │   ├── useStreamingChat.js
│       │   ├── useIngestion.js
│       │   └── useRepos.js
│       ├── services/
│       │   └── api.js
│       └── styles/
│           └── index.css
├── tests/
│   ├── golden_qa/
│   │   └── README.md
│   ├── eval_retrieval.py
│   ├── eval_generation.py
│   ├── load_test.py
│   └── conftest.py
├── chroma_data/           (gitignored — Docker volume)
├── ollama_models/         (gitignored — Docker volume)
└── repos_cache/           (gitignored — ephemeral clones)
```

---

## Port Allocation

| Service | Port | Purpose |
|---|---|---|
| frontend | 5173 | Vite dev server |
| backend | 8000 | FastAPI REST + WebSocket |
| ollama | 11434 | LLM + embedding inference |
| chroma | 8001 | ChromaDB HTTP API |

---

## API Endpoints Reference

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| POST | `/api/v1/ingest` | `{repo_url, branch}` | `{job_id, status}` | Start background ingestion |
| GET | `/api/v1/ingest/{job_id}/status` | — | SSE stream `{progress, current_file}` | Live ingestion progress |
| POST | `/api/v1/query` | `{repo_id, question, model?}` | `{answer, sources}` | Non-streaming query |
| WS | `/api/v1/ws/chat` | `{repo_id, question, model?}` | Token stream | Streaming chat |
| GET | `/api/v1/repos` | — | `[{repo_id, url, chunks, indexed_at}]` | List indexed repos |
| DELETE | `/api/v1/repos/{repo_id}` | — | `{deleted: true}` | Wipe collection |
| GET | `/api/v1/health` | — | `{status, chroma, ollama}` | Health check |

---

## PHASE 0 — Environment Bootstrap

**Goal:** Local machine ready with Ollama running and project scaffold created.

### Step 0.1 — Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify:** `ollama --version` prints a version number.

### Step 0.2 — Pull all three models

```bash
ollama pull codellama:13b
ollama pull nomic-embed-text
ollama pull llama3:8b
```

**Verify:** `curl http://localhost:11434/api/tags` returns JSON listing all 3 models.

### Step 0.3 — Create project scaffold

Create the full directory structure shown in the "Project File Structure" section above. Create every directory and every empty `__init__.py` file.

```bash
mkdir -p codebase-oracle
cd codebase-oracle
mkdir -p backend/{api,ingestion,retrieval,models,core,jobs}
mkdir -p frontend/src/{components/{Sidebar,Chat,Sources},hooks,services,styles}
mkdir -p frontend/public
mkdir -p tests/golden_qa
mkdir -p chroma_data ollama_models repos_cache
touch backend/api/__init__.py backend/ingestion/__init__.py backend/retrieval/__init__.py
touch backend/models/__init__.py backend/core/__init__.py backend/jobs/__init__.py
touch docker-compose.yml .env README.md .gitignore
```

### Step 0.4 — Create `.gitignore`

```gitignore
.env
chroma_data/
ollama_models/
repos_cache/
__pycache__/
*.pyc
node_modules/
dist/
.pytest_cache/
*.egg-info/
```

### Step 0.5 — Create `.env`

```env
GITHUB_TOKEN=
OLLAMA_URL=http://ollama:11434
CHROMA_HOST=http://chroma:8000
DEFAULT_LLM_MODEL=codellama:13b
MAX_REPO_SIZE_MB=500
LOG_LEVEL=info
BM25_CACHE_DIR=/tmp/bm25
```

### Step 0.6 — Create `backend/requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
httpx==0.27.0
chromadb==0.5.0
gitpython==3.1.43
tree-sitter==0.21.3
tree-sitter-python==0.21.0
tree-sitter-javascript==0.21.0
tree-sitter-typescript==0.21.0
tree-sitter-java==0.21.0
tree-sitter-go==0.21.0
tree-sitter-rust==0.21.0
rank-bm25==0.2.2
sentence-transformers==3.0.0
pydantic==2.7.0
python-multipart==0.0.9
```

**Verify:** `pip install -r backend/requirements.txt` completes without errors.

**PHASE 0 GATE:** Ollama is running. All 3 models are pulled. Project directory structure exists. Dependencies install cleanly.

---

## PHASE 1 — Ingestion Pipeline

**Goal:** Clone a GitHub repo, parse every source file via tree-sitter AST chunking, embed chunks, store in ChromaDB.

### Step 1.1 — Create `backend/core/config.py`

Pydantic Settings class that reads all values from `.env`:
- `OLLAMA_URL` (default `http://ollama:11434`)
- `CHROMA_HOST` (default `http://chroma:8000`)
- `DEFAULT_LLM_MODEL` (default `codellama:13b`)
- `MAX_REPO_SIZE_MB` (default `500`)
- `LOG_LEVEL` (default `info`)
- `BM25_CACHE_DIR` (default `/tmp/bm25`)
- `GITHUB_TOKEN` (default empty string)

### Step 1.2 — Create `backend/core/logger.py`

Structured logging config. Use Python's built-in `logging` module. Log format: `%(asctime)s %(levelname)s [%(name)s] %(message)s`. Read level from config.

### Step 1.3 — Create `backend/models/schemas.py`

Implement these exact Pydantic models:

```python
class IngestRequest(BaseModel):
    repo_url: HttpUrl
    branch: str = "main"

class QueryRequest(BaseModel):
    repo_id: str
    question: str
    model: Optional[str] = "codellama:13b"

class SourceChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str
```

### Step 1.4 — Create `backend/ingestion/git_cloner.py`

Write a `GitCloner` class:
- Takes a `repo_url` and `branch`
- Generates `repo_id = md5(repo_url)[:8]`
- Clones to `/tmp/repos/{repo_id}` using `gitpython` with `depth=1`
- If the directory already exists, do `git pull` instead
- Return the clone path and the repo_id
- Use the `GITHUB_TOKEN` from env if set (inject into URL for HTTPS clones)

**Verify:** Calling `clone("https://github.com/pallets/flask", "main")` clones the repo and returns a valid path.

### Step 1.5 — Create `backend/ingestion/file_walker.py`

Write a `FileWalker` class:
- Walks the cloned repo directory recursively
- Maps file extensions to languages using this exact mapping:
  ```python
  SUPPORTED_EXTENSIONS = {
      ".py": "python", ".js": "javascript", ".ts": "typescript",
      ".java": "java", ".go": "go", ".rs": "rust",
      ".md": "markdown", ".txt": "plaintext",
      ".jsx": "javascript", ".tsx": "typescript",
  }
  ```
- Skips files not in the mapping
- Skips `.git/`, `node_modules/`, `__pycache__/`, `venv/`, `.venv/` directories
- Returns list of `(file_path_relative, full_path, language)` tuples

### Step 1.6 — Create `backend/ingestion/ast_chunker.py`

Implement `ASTChunker` class exactly as specified in the blueprint:
- Constructor takes `language: str`, builds a tree-sitter parser for that language
- `chunk(source_code, file_path) -> list[dict]` method
- Each dict has: `text`, `file_path`, `language`, `chunk_type`, `start_line`, `end_line`, `symbol_name`
- `CHUNK_TOKEN_LIMIT = 512` (use `len(text.split())` as approximate token count)
- `TARGET_NODE_TYPES` dict mapping language to set of AST node types to extract:
  - python: `function_definition`, `class_definition`, `decorated_definition`
  - javascript: `function_declaration`, `arrow_function`, `class_declaration`
  - typescript: `function_declaration`, `arrow_function`, `class_declaration`
  - java: `method_declaration`, `class_declaration`
  - go: `function_declaration`, `method_declaration`
- `_traverse(node, source, file_path, out)` — recursive; captures matching nodes if under token limit
- `_extract_name(node, source)` — finds `identifier` or `name` child nodes
- `_sliding_window_fallback(source, file_path)` — for non-parseable files (markdown, plaintext), window=60 lines, stride=45 lines

**Verify:** Chunking a Python file with 3 functions produces 3 chunks, each with correct `symbol_name`, `start_line`, `end_line`.

### Step 1.7 — Create `backend/ingestion/embedding_service.py`

Implement `EmbeddingService` class:
- `embed_batch(texts: list[str]) -> list[list[float]]`
- Uses `httpx.AsyncClient` to call Ollama
- Endpoint: `{OLLAMA_URL}/api/embeddings`
- Model: `nomic-embed-text`
- Batch size: 32 (process in chunks of 32)
- For each text, call `_embed_one(client, text)` which POSTs to `/api/embeddings` with `{"model": "nomic-embed-text", "prompt": text[:8000]}`
- Use `asyncio.gather` for concurrent embedding within each batch
- Timeout: 120 seconds per batch

**Verify:** Embedding 5 strings returns 5 vectors, each of dimension 768.

### Step 1.8 — Create `backend/ingestion/chroma_writer.py`

Implement `ChromaWriter` class:
- Constructor: `chromadb.PersistentClient(path="/chroma_db")`
- `upsert(collection_name, chunks, embeddings)` method
- Creates collection with `metadata={"hnsw:space": "cosine"}`
- Generates UUID v4 IDs for each chunk
- Stores document text, embeddings, and metadata (file_path, language, chunk_type, start_line, end_line, symbol_name)

**Verify:** After upserting 10 chunks, `collection.count()` returns 10.

### Step 1.9 — Create `backend/ingestion/orchestrator.py`

Implement `IngestionOrchestrator` class exactly as specified:
- Constructor takes `ChromaWriter` and `EmbeddingService`
- `ingest_repo(repo_url, branch, progress_callback)` async method
- Generates `repo_id = md5(repo_url)[:8]`
- Clones/pulls the repo
- Walks all files, chunks each one with `ASTChunker`
- Batch embeds all chunks
- Writes to ChromaDB collection `repo_{repo_id}`
- Returns `{"repo_id", "collection", "chunks_indexed"}`
- Calls `progress_callback(current_index, total_files)` during processing

### Step 1.10 — Create `backend/ingestion/bm25_builder.py`

Implement `BM25Builder`:
- `build_index(collection_name, chroma_client) -> (BM25Okapi, list[dict])`
- Fetches all documents + metadata from the ChromaDB collection
- Tokenizes each document text (lowercase + split on whitespace)
- Builds a `BM25Okapi` index from the tokenized corpus
- Persists the tokenized corpus to `{BM25_CACHE_DIR}/{collection_name}.json`
- Returns the BM25 index and the corpus (list of chunk dicts with text + metadata)

### Step 1.11 — Integration test

Write a script or test that:
1. Clones a small public repo (e.g. `https://github.com/pallets/markupsafe`)
2. Walks files, chunks them, embeds, writes to ChromaDB
3. Builds BM25 index
4. Prints chunk count and sample chunk

**PHASE 1 GATE:** ChromaDB has records. BM25 index file exists on disk. `collection.count()` > 0 for the test repo.

---

## PHASE 2 — Hybrid Retrieval Pipeline

**Goal:** Query returns top-8 relevant code chunks using BM25 + dense + RRF + cross-encoder reranking.

### Step 2.1 — Create `backend/retrieval/dense_retriever.py`

Wrapper around ChromaDB query:
- `retrieve(query_embedding, collection, k=50) -> list[dict]`
- Calls `collection.query(query_embeddings=[embedding], n_results=k, include=["documents","metadatas","distances"])`
- Formats results into list of dicts with `text`, metadata fields, and `score = 1 - distance`

### Step 2.2 — Create `backend/retrieval/sparse_retriever.py`

Wrapper around BM25:
- `retrieve(query, bm25_index, corpus, k=50) -> list[(score, chunk)]`
- Tokenizes query: `query.lower().split()`
- Gets BM25 scores, returns top-k by score

### Step 2.3 — Create `backend/retrieval/rrf_fusion.py`

Reciprocal Rank Fusion implementation:
- `merge(dense_chunks, sparse_scored_chunks, k=60) -> list[dict]`
- For each chunk in both lists, compute RRF score: `1 / (rank + k)`
- Deduplicate by `chunk["text"][:100]`
- Sum scores for chunks appearing in both lists
- Return sorted by total RRF score, top 100

### Step 2.4 — Create `backend/retrieval/reranker.py`

Cross-encoder reranker:
- Lazy-load `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)` on first call
- `rerank(query, candidates, top_k=8) -> list[dict]`
- Creates `(query, chunk_text)` pairs
- Calls `cross_encoder.predict(pairs)`
- Returns top-k sorted by cross-encoder score

### Step 2.5 — Create `backend/retrieval/hybrid_retriever.py`

Implement `HybridRetriever` class exactly as specified:
- Constructor takes `collection_name`, `chroma_client`, `embed_service`, `bm25_index`, `corpus`
- `retrieve(query, final_k=8) -> list[dict]` async method
- Steps:
  1. Embed query via `embed_service`
  2. Dense retrieval: top-50 from ChromaDB
  3. Sparse retrieval: top-50 from BM25
  4. RRF fusion of both lists
  5. Cross-encoder rerank: top-100 -> top-8
- Return final top-8 chunks

### Step 2.6 — Unit test retrieval quality

Test against the repo indexed in Phase 1:
- Pick 5 known functions/classes in the repo
- Query for each by name and by description
- Verify the expected file appears in the top-5 results

**PHASE 2 GATE:** Hybrid retrieval returns relevant chunks for known queries. Both BM25 and dense paths contribute results.

---

## PHASE 3 — LLM Generation Layer

**Goal:** Send retrieved chunks + user question to Ollama, stream the answer back.

### Step 3.1 — Create `backend/retrieval/prompt_builder.py`

Implement `build_prompt(query, chunks) -> list[dict]` exactly as specified:
- System prompt: "You are CodeBase Oracle..." with rules about citing file paths, not hallucinating, and admitting when context is insufficient
- User message: formatted code chunks with `[Chunk N] File: path Lines X-Y (type)` headers, each in a markdown code block
- Returns messages list for Ollama `/api/chat`

### Step 3.2 — Create `backend/retrieval/ollama_client.py`

Implement `stream_response(prompt_messages, model) -> AsyncGenerator[str, None]`:
- POST to `{OLLAMA_URL}/api/chat` with `stream: True`
- Options: `temperature: 0.1`, `num_ctx: 8192`, `top_p: 0.9`
- Uses `httpx.AsyncClient.stream()` with 300s timeout
- Yields each token from the stream
- Stops when `data.get("done")` is True

### Step 3.3 — Implement context window guard

Before calling the LLM:
- Count approximate tokens in the full prompt (sum of all chunk text lengths / 4)
- If over 4096 tokens (for 7B models) or 8192 (for 13B), trim chunks from the end until under limit

### Step 3.4 — Implement hallucination guardrail

```python
MIN_SIMILARITY_THRESHOLD = 0.25

def has_sufficient_context(chunks: list[dict]) -> bool:
    if not chunks:
        return False
    best_score = max(c.get("score", 0) for c in chunks)
    return best_score >= MIN_SIMILARITY_THRESHOLD
```

If `has_sufficient_context` returns False, return a canned response: "I don't have enough context in the indexed codebase to answer this confidently." — do NOT call the LLM.

### Step 3.5 — End-to-end test (no API yet)

Write a script that:
1. Loads the indexed repo from Phase 1
2. Runs hybrid retrieval for a question
3. Checks the guardrail
4. Builds the prompt
5. Streams the response from Ollama
6. Prints the full answer with source citations

**PHASE 3 GATE:** Ollama returns a streamed answer that references actual files and line numbers from the indexed repo.

---

## PHASE 4 — FastAPI Backend

**Goal:** All endpoints working, CORS configured, structured logging enabled.

### Step 4.1 — Create `backend/core/dependencies.py`

FastAPI dependency injection functions:
- `get_chroma_client()` — returns `chromadb.PersistentClient(path="/chroma_db")`
- `get_embed_service()` — returns `EmbeddingService()` instance
- `get_settings()` — returns the `Settings` config object

### Step 4.2 — Create `backend/jobs/job_store.py`

In-memory job tracking:
```python
# Dict: {job_id: {"status": str, "progress": float, "current_file": str, "error": str|None}}
```
- `create_job(job_id)` — sets status to "running", progress to 0
- `update_job(job_id, progress, current_file)` — updates progress
- `complete_job(job_id)` — sets status to "completed", progress to 1.0
- `fail_job(job_id, error)` — sets status to "failed"
- `get_job(job_id)` — returns job dict or None

### Step 4.3 — Create `backend/api/routes_ingest.py`

- `POST /api/v1/ingest` — accepts `IngestRequest`, generates UUID `job_id`, starts ingestion in `BackgroundTasks`, returns `{"job_id": ..., "status": "running"}`
- `GET /api/v1/ingest/{job_id}/status` — SSE endpoint that yields job progress every 1 second until job completes or fails

### Step 4.4 — Create `backend/api/routes_query.py`

- `POST /api/v1/query` — accepts `QueryRequest`, runs hybrid retrieval, checks guardrail, builds prompt, calls Ollama (non-streaming), returns `QueryResponse`
- `WS /api/v1/ws/chat` — WebSocket endpoint:
  - Receives JSON `{repo_id, question, model}`
  - Runs retrieval
  - Sends `{"type": "sources", "sources": [...]}` with chunk metadata
  - Streams `{"type": "token", "token": "..."}` for each LLM token
  - Sends `{"type": "done"}` when complete

### Step 4.5 — Create `backend/api/routes_repos.py`

- `GET /api/v1/repos` — lists all ChromaDB collections, returns repo info
- `DELETE /api/v1/repos/{repo_id}` — deletes the ChromaDB collection and BM25 cache file
- `GET /api/v1/health` — checks Ollama and ChromaDB connectivity, returns status

### Step 4.6 — Create `backend/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CodeBase Oracle API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all three routers with prefix="/api/v1"
```

### Step 4.7 — Test all endpoints

Using `curl` or a REST client:
1. `POST /api/v1/ingest` with a small repo URL → get `job_id`
2. `GET /api/v1/ingest/{job_id}/status` → see progress stream
3. `GET /api/v1/repos` → see the repo listed
4. `POST /api/v1/query` with a question → get answer + sources
5. Test WebSocket `/api/v1/ws/chat` → receive streaming tokens
6. `GET /api/v1/health` → all services "ok"
7. `DELETE /api/v1/repos/{repo_id}` → collection removed

**PHASE 4 GATE:** All 7 endpoints return correct responses. SSE streaming works. WebSocket streaming works.

---

## PHASE 5 — React Frontend

**Goal:** Working UI with repo management, chat with streaming, and source code cards.

### Step 5.1 — Scaffold Vite + React app

```bash
cd frontend
npm create vite@latest . -- --template react
npm install
```

Create `vite.config.js` with proxy for `/api` to `http://localhost:8000` and WebSocket passthrough.

### Step 5.2 — Create `frontend/src/services/api.js`

Fetch/axios wrappers for all REST endpoints:
- `ingestRepo(repoUrl, branch)` → POST `/api/v1/ingest`
- `getRepos()` → GET `/api/v1/repos`
- `deleteRepo(repoId)` → DELETE `/api/v1/repos/{repoId}`
- `queryRepo(repoId, question, model)` → POST `/api/v1/query`
- `getHealth()` → GET `/api/v1/health`

### Step 5.3 — Create `frontend/src/hooks/useIngestion.js`

SSE hook:
- Connects to `/api/v1/ingest/{jobId}/status`
- Parses SSE events
- Exposes `{progress, currentFile, status, error}`

### Step 5.4 — Create `frontend/src/hooks/useStreamingChat.js`

WebSocket hook exactly as specified in the blueprint:
- Opens WebSocket to `ws://localhost:8000/api/v1/ws/chat`
- On open: sends `{repo_id, question, model}` as JSON
- On message: handles `token`, `sources`, and `done` event types
- Accumulates tokens into assistant message
- Exposes `{messages, sendMessage}`

### Step 5.5 — Create `frontend/src/hooks/useRepos.js`

- Fetches repo list on mount
- Provides `deleteRepo` action
- Auto-refreshes after ingest completes

### Step 5.6 — Create Sidebar components

**RepoManager.jsx:**
- Text input for GitHub URL
- Text input for branch (default "main")
- "Index Repo" button that calls `ingestRepo`
- Shows `IngestionProgress` while running

**IngestionProgress.jsx:**
- Progress bar driven by SSE
- Shows current file being processed
- Shows "Complete" or error message

**IndexedRepoList.jsx:**
- Cards showing: repo URL, chunk count, last indexed time
- Delete button per repo

### Step 5.7 — Create Chat components

**ChatWindow.jsx:**
- Scrollable message thread
- Contains `QueryInput` at bottom

**QueryInput.jsx:**
- Textarea for question
- Model selector dropdown (CodeLlama 13B / Llama3 8B)
- Send button
- Must select a repo before sending

**UserMessage.jsx / AssistantMessage.jsx:**
- User: right-aligned bubble
- Assistant: left-aligned, renders markdown, shows SourceCards below

**StreamingCursor.jsx:**
- Blinking cursor indicator during token streaming

### Step 5.8 — Create Source components

**SourceCards.jsx:**
- Horizontal scrollable row of code snippet cards

**CodeSnippetCard.jsx:**
- Shows: file path, line range, language tag
- Collapsible code preview with syntax highlighting
- Displays chunk_type badge

### Step 5.9 — Create `frontend/src/App.jsx`

Layout: Sidebar (left, ~300px) + ChatWindow (right, flex-grow).

### Step 5.10 — Style with CSS

Apply clean, modern styling. Use Tailwind CSS or plain CSS. Dark theme preferred for a code-focused tool.

**PHASE 5 GATE:** User can paste a repo URL, click "Index", see progress, then ask questions and see streaming answers with source code cards.

---

## PHASE 6 — Docker Compose Stack

**Goal:** `docker compose up --build` brings up all 5 services from cold start.

### Step 6.1 — Create `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Step 6.2 — Create `frontend/Dockerfile`

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

### Step 6.3 — Create `docker-compose.yml`

5 services exactly as specified in the blueprint:

1. **ollama** — `ollama/ollama:latest`, port 11434, volume `./ollama_models:/root/.ollama`, healthcheck using `curl http://localhost:11434/api/tags`
2. **model-puller** — depends on ollama healthy, pulls `nomic-embed-text`, `codellama:13b`, `llama3:8b`, restart: "no"
3. **chroma** — `chromadb/chroma:latest`, port 8001→8000, volume `./chroma_data:/chroma_db`, CORS config, healthcheck using `/api/v1/heartbeat`
4. **backend** — builds from `./backend`, port 8000, mounts `chroma_data` and `repos_cache`, env vars for OLLAMA_URL/CHROMA_HOST, depends on ollama + chroma healthy, healthcheck on `/api/v1/health`
5. **frontend** — builds from `./frontend`, port 5173, env vars for API/WS base URLs, depends on backend healthy

### Step 6.4 — Test cold start

```bash
docker compose up --build
```

Wait for all healthchecks to pass. Then test:
1. `curl http://localhost:8000/api/v1/health`
2. Open `http://localhost:5173` in browser
3. Index a repo through the UI
4. Ask a question and get a streamed answer

**PHASE 6 GATE:** `docker compose up --build` from an empty state (no volumes) pulls models, starts all services, and serves the full application.

---

## PHASE 7 — Testing & Evaluation

**Goal:** Quantified retrieval quality and working eval harness.

### Step 7.1 — Create golden QA test set

In `tests/golden_qa/`, create a JSON file for a test repo with 20-30 questions:

```json
[
  {
    "question": "...",
    "expected_file": "path/to/file.py",
    "expected_symbol": "function_name",
    "answer_keywords": ["keyword1", "keyword2"]
  }
]
```

### Step 7.2 — Create `tests/eval_retrieval.py`

Implement `evaluate_retrieval(test_set, retriever, k=5)` exactly as specified:
- For each question, run retrieval
- Check if `expected_file` or `expected_symbol` appear in results
- Compute Recall@K and MRR
- Print results

Target: Recall@5 > 0.70, MRR > 0.60

### Step 7.3 — Create `tests/eval_generation.py`

LLM-as-Judge evaluation:
- For each QA pair, generate an answer
- Check if `answer_keywords` appear in the response
- Compute faithfulness score

Target: > 0.80

### Step 7.4 — Create `tests/load_test.py`

Locust script for concurrent query simulation:
- Target: p95 query latency < 8 seconds on GPU

### Step 7.5 — Create `tests/conftest.py`

Pytest fixtures:
- Temporary ChromaDB collection
- Pre-loaded test repo
- Embedding service mock (optional, for fast unit tests)

### Step 7.6 — Run evaluations and iterate

1. Run eval on your test repo
2. If Recall@5 < 0.70: tune chunk sizes, BM25 weights, reranker
3. If faithfulness < 0.80: tune system prompt, temperature
4. Document final metrics in README

**PHASE 7 GATE:** Eval scripts run without errors. Metrics are documented. Top retrieval failures are identified and fixed if possible.

---

## PHASE 8 — Security Hardening

**Goal:** Production-ready security posture for local deployment.

### Step 8.1 — Input validation

- All endpoints use Pydantic schemas (already done in Phase 4)
- Reject non-HTTPS repo URLs
- Reject URLs not matching `github.com` or other allowed hosts

### Step 8.2 — Rate limiting

Install `slowapi`. Add rate limit to POST `/ingest` (e.g. 5 requests per minute).

### Step 8.3 — Repo size guard

Before cloning, check repo size. Reject repos > `MAX_REPO_SIZE_MB`. Use GitHub API or `git ls-remote` to estimate.

### Step 8.4 — Token security

- `.env` file is in `.gitignore`
- `GITHUB_TOKEN` is read from env, never logged
- CORS restricted to `localhost:5173` only

### Step 8.5 — Error handling

- All endpoints return structured error responses
- No stack traces leaked to client in production
- Background ingestion errors captured in job store

**PHASE 8 GATE:** No secrets in code. Rate limiting active. Size guard rejects oversized repos. Error responses are clean.

---

## PHASE 9 — README & Documentation

### Step 9.1 — Write `README.md`

Sections:
1. What is CodeBase Oracle (one paragraph)
2. Architecture diagram (ASCII from blueprint)
3. Prerequisites (Docker, Ollama, GPU recommendations)
4. Quick Start: `docker compose up --build`
5. First ingest: POST request example
6. First query: POST request example
7. Environment variables reference
8. Hardware requirements table
9. Troubleshooting common issues
10. Research citations

### Step 9.2 — Write `tests/golden_qa/README.md`

How to add new test cases for new repos.

**PHASE 9 GATE:** README fully documents the project. A new user can go from zero to running by following the README alone.

---

## Completion Checklist

After all phases, verify:

- [ ] `docker compose up --build` works from scratch
- [ ] Ollama models load successfully
- [ ] Can index a public GitHub repo from the UI
- [ ] Ingestion progress streams via SSE
- [ ] Can ask questions and get streamed answers
- [ ] Source cards show correct file paths and line numbers
- [ ] Can switch between CodeLlama and Llama3 models
- [ ] Can delete an indexed repo
- [ ] Health endpoint reports all services OK
- [ ] Eval scripts produce Recall@5 and MRR numbers
- [ ] No secrets committed to git
- [ ] README is complete and accurate

---

*Plan generated from codebase_oracle_blueprint (1).md — v1.1, June 2026*
