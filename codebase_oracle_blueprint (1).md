# CodeBase Oracle — RAG-Powered Dev Assistant for GitHub Repos
## Production-Ready Technical Blueprint

> **Audience:** CSE engineer with working knowledge of Python, FastAPI, Docker, and basic ML concepts.  
> **Goal:** Zero-to-running, locally-hosted, zero-API-cost codebase chat assistant backed by state-of-the-art RAG.

---

## Table of Contents

1. [System Overview & Design Philosophy](#1-system-overview--design-philosophy)
2. [Step-by-Step Development Plan](#2-step-by-step-development-plan)
3. [Compute Architecture & Infrastructure](#3-compute-architecture--infrastructure)
4. [Ingestion Pipeline — Code-Aware Chunking](#4-ingestion-pipeline--code-aware-chunking)
5. [Embedding & Vector Store Strategy](#5-embedding--vector-store-strategy)
6. [Hybrid Retrieval Pipeline](#6-hybrid-retrieval-pipeline)
7. [LLM Inference Layer](#7-llm-inference-layer)
8. [FastAPI Backend — API Contract](#8-fastapi-backend--api-contract)
9. [React/Vite Frontend](#9-reactvite-frontend)
10. [Docker Compose Stack](#10-docker-compose-stack)
11. [Security & Data Sovereignty Checklist](#11-security--data-sovereignty-checklist)
12. [Testing & Evaluation Framework](#12-testing--evaluation-framework)
13. [Phased Roadmap (Weeks)](#13-phased-roadmap-weeks)
14. [File System Architecture](#14-file-system-architecture)

---

## 1. System Overview & Design Philosophy

### What You're Building

```
GitHub Repo URL
      │
      ▼
[Ingestion Service]  ──tree-sitter AST chunking──►  [ChromaDB]  ◄── nomic-embed-text (Ollama)
                                                         │
                                                         ▼
User Query ──► [FastAPI /query] ──► [Hybrid Retriever] ──► [Reranker] ──► [Ollama LLM] ──► Answer
                                      BM25 + Dense                        Llama3 / CodeLlama
```

### Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Chunking strategy | AST-based via tree-sitter | 70.1% Recall@5 vs 42.4% for fixed-size (cAST paper, 2025) |
| Retrieval | Hybrid BM25 + Dense | Recall improves from ~0.72 (BM25 alone) to ~0.91 (hybrid) |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 | +35–40% answer accuracy vs no reranking; runs locally in <50ms |
| Embedding | nomic-embed-text (Ollama) | 8192 token context, code-aware, fully local |
| LLM | CodeLlama 13B or Llama3 8B (Ollama) | Code-specific reasoning; runs on 12GB VRAM or CPU fallback |
| Vector DB | ChromaDB persistent | Simple, file-based, no extra infra |
| API | FastAPI + WebSocket | Streaming token output for UX |
| Containerisation | Docker Compose | 5-service stack, one command startup |

---

## 2. Step-by-Step Development Plan

### Phase 0 — Environment Bootstrap (Day 1)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull codellama:13b          # primary code LLM
ollama pull nomic-embed-text       # embedding model
ollama pull llama3:8b              # fallback/chat model

# 3. Verify Ollama is live
curl http://localhost:11434/api/tags

# 4. Create project scaffold
mkdir codebase-oracle && cd codebase-oracle
mkdir -p backend/{ingestion,retrieval,api,models} frontend chroma_data
touch docker-compose.yml .env README.md
```

### Phase 1 — Ingestion Pipeline (Days 2–4)

**Goal:** Clone a GitHub repo → parse every source file → AST-chunk it → embed → store in ChromaDB.

```
Step 1.1  Install gitpython + tree-sitter + language grammars
Step 1.2  Write GitCloner: clone/pull repo to a temp dir, walk file tree
Step 1.3  Write ASTChunker: detect language, parse with tree-sitter, recursive split-merge
Step 1.4  Write EmbeddingService: batch-embed chunks via Ollama nomic-embed-text
Step 1.5  Write ChromaWriter: upsert chunks with metadata (file_path, lang, chunk_type, start_line, end_line)
Step 1.6  Write ingestion FastAPI endpoint POST /ingest  { repo_url, branch }
Step 1.7  Test end-to-end: point at a small public repo, verify ChromaDB has records
```

### Phase 2 — Hybrid Retrieval (Days 5–7)

```
Step 2.1  Build BM25 index (rank_bm25 library) from stored chunk texts at query time or cache on disk
Step 2.2  Build DenseRetriever: embed query → ChromaDB cosine similarity search (top-50)
Step 2.3  Build HybridFuser: Reciprocal Rank Fusion (RRF) of BM25 + dense scores
Step 2.4  Integrate cross-encoder reranker (sentence-transformers): top-50 → top-8
Step 2.5  Unit-test retrieval quality on known queries against an indexed repo
```

### Phase 3 — LLM Generation (Days 8–9)

```
Step 3.1  Build PromptBuilder: system prompt + retrieved chunks (with file/line provenance) + user query
Step 3.2  Build OllamaClient: streaming POST to /api/generate, yield tokens to FastAPI
Step 3.3  Implement context window guard: count tokens, trim if > 4096 for 7B or > 8192 for 13B
Step 3.4  Add hallucination guardrail: if similarity distance > threshold, reply "not enough context"
```

### Phase 4 — FastAPI Backend (Days 10–11)

```
Step 4.1  Wire all modules into FastAPI app
Step 4.2  POST /ingest    — trigger ingestion job
Step 4.3  GET  /status    — ingestion progress (SSE stream)
Step 4.4  POST /query     — hybrid retrieve + rerank + generate
Step 4.5  WS   /ws/chat   — WebSocket for streaming tokens
Step 4.6  GET  /repos     — list indexed repos + chunk counts
Step 4.7  DELETE /repo/{id} — wipe collection
Step 4.8  Add CORS, request validation (Pydantic), structured logging
```

### Phase 5 — React Frontend (Days 12–14)

```
Step 5.1  Scaffold Vite + React app
Step 5.2  RepoPanel: URL input + branch selector + ingest button + progress bar
Step 5.3  ChatPanel: message thread + streaming token rendering
Step 5.4  SourcePanel: collapsible code snippet cards per retrieved chunk (file + line range)
Step 5.5  ModelSelector: switch Llama3 / CodeLlama from UI
Step 5.6  Connect WebSocket for streaming responses
```

### Phase 6 — Containerisation (Day 15)

```
Step 6.1  Write Dockerfiles for backend and frontend
Step 6.2  Write docker-compose.yml (5 services: ollama, chroma, backend, frontend, model-puller)
Step 6.3  Add volume mounts: chroma_data, ollama_models
Step 6.4  Add healthchecks + depends_on ordering
Step 6.5  Test: docker compose up --build from cold start
```

---

## 3. Compute Architecture & Infrastructure

### Service Map

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                   │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐   ┌───────────────┐  │
│  │   frontend   │    │   backend    │   │    ollama     │  │
│  │  React/Vite  │───►│   FastAPI    │──►│  :11434       │  │
│  │  :5173       │    │  :8000       │   │  codellama    │  │
│  └──────────────┘    └──────┬───────┘   │  nomic-embed  │  │
│                             │           └───────────────┘  │
│                             ▼                               │
│                    ┌──────────────┐                         │
│                    │   chroma     │                         │
│                    │  ChromaDB    │                         │
│                    │  :8001       │                         │
│                    │  /chroma_data│                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Minimum Hardware Requirements

| Tier | CPU | RAM | GPU | LLM Model |
|---|---|---|---|---|
| Minimum | 8-core | 16 GB | None | Llama3:8b (CPU, slow ~2 tok/s) |
| Recommended | 8-core | 32 GB | RTX 3080 / 4060 (10–12 GB VRAM) | CodeLlama:13b (GPU, ~15 tok/s) |
| Ideal | 16-core | 64 GB | RTX 4090 / A10 (24 GB VRAM) | CodeLlama:34b (GPU, ~12 tok/s) |

**Embedding is cheap:** nomic-embed-text runs in ~200ms/batch on CPU. Keep Ollama's embed endpoint separate from generation to avoid GPU contention.

### Port Allocation

| Service | Port | Purpose |
|---|---|---|
| frontend | 5173 | Vite dev server |
| backend | 8000 | FastAPI REST + WS |
| ollama | 11434 | LLM + embed inference |
| chroma | 8001 | Vector DB HTTP API |

### File System Volumes

```
./chroma_data/     → ChromaDB persistent storage (mount to /chroma_db inside container)
./ollama_models/   → Ollama model weights (avoid re-downloading on restart)
./repos_cache/     → Cloned repos (ephemeral, can be tmpfs)
```

---

## 4. Ingestion Pipeline — Code-Aware Chunking

### Why AST Chunking, Not Fixed-Size

Fixed-size chunkers split code mid-function, mid-class, or mid-comment block — destroying semantic meaning. Research from the cAST paper (2025) demonstrates AST-based recursive chunking achieves **70.1% Recall@5 vs 42.4% for fixed-size** splitting, because chunk boundaries align with real syntactic units (functions, classes, methods).

### Language Support Matrix

| Language | tree-sitter Grammar | Node Types to Extract |
|---|---|---|
| Python | tree-sitter-python | function_definition, class_definition, decorated_definition |
| JavaScript / TypeScript | tree-sitter-javascript / tsx | function_declaration, arrow_function, class_declaration |
| Java | tree-sitter-java | method_declaration, class_declaration |
| Go | tree-sitter-go | function_declaration, method_declaration |
| Rust | tree-sitter-rust | function_item, impl_item |
| C / C++ | tree-sitter-c / cpp | function_definition |

### ASTChunker Algorithm

```python
# backend/ingestion/ast_chunker.py

from tree_sitter import Language, Parser
import tree_sitter_python as tspython   # etc per language

CHUNK_TOKEN_LIMIT = 512   # sweet spot; quality drops above ~2500 tokens
CHUNK_OVERLAP_LINES = 3   # preserve surrounding context

TARGET_NODE_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "arrow_function", "class_declaration"},
    "typescript": {"function_declaration", "arrow_function", "class_declaration"},
    "java": {"method_declaration", "class_declaration"},
    "go": {"function_declaration", "method_declaration"},
}

class ASTChunker:
    def __init__(self, language: str):
        self.language = language
        self.parser = self._build_parser(language)

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        """
        Returns list of dicts:
          { text, file_path, language, chunk_type, start_line, end_line, symbol_name }
        """
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        root = tree.root_node
        chunks = []
        self._traverse(root, source_code, file_path, chunks)

        # Fallback: if no AST nodes found (e.g. config/markdown), sliding window
        if not chunks:
            chunks = self._sliding_window_fallback(source_code, file_path)
        return chunks

    def _traverse(self, node, source: str, file_path: str, out: list):
        target_types = TARGET_NODE_TYPES.get(self.language, set())
        if node.type in target_types:
            text = source[node.start_byte:node.end_byte]
            if len(text.split()) <= CHUNK_TOKEN_LIMIT:
                out.append({
                    "text": text,
                    "file_path": file_path,
                    "language": self.language,
                    "chunk_type": node.type,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "symbol_name": self._extract_name(node, source),
                })
                return   # don't recurse into already-captured node
        for child in node.children:
            self._traverse(child, source, file_path, out)

    def _extract_name(self, node, source: str) -> str:
        for child in node.children:
            if child.type in ("identifier", "name"):
                return source[child.start_byte:child.end_byte]
        return "anonymous"

    def _sliding_window_fallback(self, source: str, file_path: str) -> list[dict]:
        lines = source.splitlines()
        chunks = []
        window, stride = 60, 45   # lines
        for i in range(0, len(lines), stride):
            block = "\n".join(lines[i:i + window])
            chunks.append({
                "text": block, "file_path": file_path,
                "language": "plaintext", "chunk_type": "sliding_window",
                "start_line": i + 1, "end_line": min(i + window, len(lines)),
                "symbol_name": "",
            })
        return chunks
```

### Ingestion Orchestrator

```python
# backend/ingestion/orchestrator.py

import git, os, uuid, hashlib
from pathlib import Path
from .ast_chunker import ASTChunker
from .embedding_service import EmbeddingService
from .chroma_writer import ChromaWriter

SUPPORTED_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".md": "markdown", ".txt": "plaintext",
    ".jsx": "javascript", ".tsx": "typescript",
}

class IngestionOrchestrator:
    def __init__(self, chroma_writer: ChromaWriter, embed_service: EmbeddingService):
        self.writer = chroma_writer
        self.embedder = embed_service

    async def ingest_repo(self, repo_url: str, branch: str = "main",
                          progress_callback=None) -> dict:
        repo_id = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        clone_path = Path(f"/tmp/repos/{repo_id}")

        # Clone or pull
        if clone_path.exists():
            repo = git.Repo(clone_path)
            repo.remotes.origin.pull()
        else:
            repo = git.Repo.clone_from(repo_url, clone_path, branch=branch, depth=1)

        all_chunks = []
        files = list(clone_path.rglob("*"))
        for i, f in enumerate(files):
            if f.suffix not in SUPPORTED_EXTENSIONS or not f.is_file():
                continue
            lang = SUPPORTED_EXTENSIONS[f.suffix]
            try:
                source = f.read_text(errors="ignore")
                chunker = ASTChunker(lang)
                chunks = chunker.chunk(source, str(f.relative_to(clone_path)))
                all_chunks.extend(chunks)
            except Exception as e:
                continue  # skip unparseable files
            if progress_callback:
                await progress_callback(i, len(files))

        # Batch embed
        texts = [c["text"] for c in all_chunks]
        embeddings = await self.embedder.embed_batch(texts)

        # Write to ChromaDB
        collection_name = f"repo_{repo_id}"
        await self.writer.upsert(collection_name, all_chunks, embeddings)

        return {"repo_id": repo_id, "collection": collection_name,
                "chunks_indexed": len(all_chunks)}
```

---

## 5. Embedding & Vector Store Strategy

### Embedding Model: nomic-embed-text

**Why nomic-embed-text over all-MiniLM-L6-v2:**
- 8192 token context window (vs 512 for MiniLM) — fits entire functions without truncation
- Trained on code + text data — better semantic understanding of identifiers
- Runs natively in Ollama — no separate Python server needed
- 768-dim output — richer representation than MiniLM's 384-dim

```python
# backend/ingestion/embedding_service.py

import httpx, asyncio

OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32

class EmbeddingService:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                tasks = [self._embed_one(client, t) for t in batch]
                batch_embeddings = await asyncio.gather(*tasks)
                embeddings.extend(batch_embeddings)
        return embeddings

    async def _embed_one(self, client, text: str) -> list[float]:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:8000]}   # guard truncation
        )
        return resp.json()["embedding"]
```

### ChromaDB Writer

```python
# backend/ingestion/chroma_writer.py

import chromadb, uuid

class ChromaWriter:
    def __init__(self, persist_dir: str = "/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

    async def upsert(self, collection_name: str,
                     chunks: list[dict], embeddings: list[list[float]]):
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}   # cosine similarity
        )
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "file_path": c["file_path"],
            "language": c["language"],
            "chunk_type": c["chunk_type"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "symbol_name": c["symbol_name"],
        } for c in chunks]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
```

### ChromaDB Collection Strategy

- One collection per repo: `repo_{md5(url)[:8]}`
- `hnsw:space = cosine` — better for semantic text similarity than L2
- Metadata fields indexed for filtering: `language`, `file_path`, `chunk_type`
- Content-addressed IDs (UUID v4) prevent duplicates on re-index

---

## 6. Hybrid Retrieval Pipeline

### Why Hybrid (BM25 + Dense)?

- BM25 excels at **exact identifier matches**: `WebSocketHandler`, `MongoClient.connect`, `init_db`
- Dense embeddings excel at **semantic queries**: "how is the database initialized?"
- Combined via Reciprocal Rank Fusion (RRF): recall improves from ~0.72 (BM25) to ~0.91 (hybrid)

### Retrieval Flow

```
Query: "how does WebSocket authentication work?"
        │
        ├──► BM25 index ──────────────────────── top-50 lexical candidates
        │
        └──► ChromaDB cosine search ──────────── top-50 dense candidates
                                │
                                ▼
                         RRF Fusion: merge + re-score both lists
                                │
                                ▼
                   cross-encoder reranker → top-8 chunks
                                │
                                ▼
                       Prompt Builder → LLM
```

### Hybrid Retriever Implementation

```python
# backend/retrieval/hybrid_retriever.py

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import chromadb, numpy as np

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

class HybridRetriever:
    def __init__(self, collection_name: str, chroma_client,
                 embed_service, bm25_index=None, corpus=None):
        self.collection = chroma_client.get_collection(collection_name)
        self.embedder = embed_service
        self.bm25 = bm25_index      # pre-built from corpus
        self.corpus = corpus        # list of chunk dicts (text, metadata)

    async def retrieve(self, query: str, final_k: int = 8) -> list[dict]:
        candidate_k = 50

        # --- Dense retrieval ---
        query_embedding = await self.embedder._embed_one(None, query)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )
        dense_chunks = self._format_chroma_results(dense_results)

        # --- Sparse BM25 retrieval ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_chunks = [self.corpus[i] for i in top_bm25_idx]
        bm25_scored = list(zip(bm25_scores[top_bm25_idx], bm25_chunks))

        # --- RRF Fusion ---
        candidates = self._rrf_merge(dense_chunks, bm25_scored)

        # --- Cross-encoder reranking → top-k ---
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:final_k]]

    def _rrf_merge(self, dense: list, sparse: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, chunk in enumerate(dense):
            key = chunk["text"][:100]
            scores[key] = scores.get(key, {"chunk": chunk, "score": 0})
            scores[key]["score"] += 1 / (rank + k)
        for rank, (_, chunk) in enumerate(sparse):
            key = chunk["text"][:100]
            if key not in scores:
                scores[key] = {"chunk": chunk, "score": 0}
            scores[key]["score"] += 1 / (rank + k)
        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [m["chunk"] for m in merged[:100]]   # pass top-100 to reranker

    def _format_chroma_results(self, results) -> list[dict]:
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({"text": doc, **meta, "score": 1 - dist})
        return chunks
```

### BM25 Index Refresh Strategy

- Build BM25 in memory after ingestion completes (rank_bm25 library)
- Persist tokenized corpus to disk as JSON for fast reload on backend restart
- Re-index automatically when a repo is re-ingested
- Health check endpoint compares ChromaDB doc count vs BM25 corpus size to detect staleness

---

## 7. LLM Inference Layer

### Prompt Engineering for Code QA

```python
# backend/retrieval/prompt_builder.py

SYSTEM_PROMPT = """You are CodeBase Oracle, an expert software engineer assistant.
You answer questions about a specific GitHub codebase ONLY using the provided code chunks.
Rules:
- Always cite the file path and line number(s) when referencing code.
- If the retrieved chunks do not contain enough information, say "I don't have enough context in the indexed codebase to answer this confidently."
- Do NOT hallucinate function names, class names, or file paths.
- Format code examples in markdown code blocks with the correct language tag.
"""

def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1}] File: `{c['file_path']}` "
            f"Lines {c['start_line']}–{c['end_line']} ({c['chunk_type']})\n"
            f"```{c['language']}\n{c['text']}\n```"
        )
    context = "\n\n---\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from codebase:\n{context}\n\nQuestion: {query}"}
    ]
```

### Ollama Streaming Client

```python
# backend/retrieval/ollama_client.py

import httpx, json
from typing import AsyncGenerator

OLLAMA_URL = "http://ollama:11434"
DEFAULT_MODEL = "codellama:13b"

async def stream_response(prompt_messages: list[dict],
                           model: str = DEFAULT_MODEL) -> AsyncGenerator[str, None]:
    payload = {
        "model": model,
        "messages": prompt_messages,
        "stream": True,
        "options": {
            "temperature": 0.1,        # low temp for factual code QA
            "num_ctx": 8192,           # context window
            "top_p": 0.9,
        }
    }
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat",
                                  json=payload) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("message", {}).get("content"):
                        yield token
                    if data.get("done"):
                        break
```

### Hallucination Guardrail

```python
# In retrieval pipeline, before calling LLM
MIN_SIMILARITY_THRESHOLD = 0.25  # cosine similarity

def has_sufficient_context(chunks: list[dict]) -> bool:
    if not chunks:
        return False
    best_score = max(c.get("score", 0) for c in chunks)
    return best_score >= MIN_SIMILARITY_THRESHOLD
```

---

## 8. FastAPI Backend — API Contract

### Project Structure

```
backend/
├── main.py
├── ingestion/
│   ├── orchestrator.py
│   ├── ast_chunker.py
│   ├── embedding_service.py
│   └── chroma_writer.py
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── prompt_builder.py
│   └── ollama_client.py
├── api/
│   ├── routes_ingest.py
│   ├── routes_query.py
│   └── routes_repos.py
└── models/
    └── schemas.py
```

### Endpoints

```python
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_ingest import router as ingest_router
from api.routes_query import router as query_router
from api.routes_repos import router as repos_router

app = FastAPI(title="CodeBase Oracle API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(ingest_router, prefix="/api/v1")
app.include_router(query_router,  prefix="/api/v1")
app.include_router(repos_router,  prefix="/api/v1")
```

### API Route Reference

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| POST | `/api/v1/ingest` | `{repo_url, branch}` | `{job_id, status}` | Kick off background ingestion |
| GET | `/api/v1/ingest/{job_id}/status` | — | SSE stream `{progress, current_file}` | Live ingestion progress |
| POST | `/api/v1/query` | `{repo_id, question, model?}` | `{answer, sources}` | Non-streaming query |
| WS | `/api/v1/ws/chat` | `{repo_id, question, model?}` | Token stream | Streaming chat |
| GET | `/api/v1/repos` | — | `[{repo_id, url, chunks, indexed_at}]` | List indexed repos |
| DELETE | `/api/v1/repos/{repo_id}` | — | `{deleted: true}` | Wipe collection |
| GET | `/api/v1/health` | — | `{status, chroma, ollama}` | Health check |

### Pydantic Schemas

```python
# backend/models/schemas.py

from pydantic import BaseModel, HttpUrl
from typing import Optional

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

---

## 9. React/Vite Frontend

### Component Architecture

```
App
├── Sidebar
│   ├── RepoManager
│   │   ├── RepoURLInput
│   │   ├── BranchSelector
│   │   ├── IngestButton
│   │   └── IngestionProgress
│   └── IndexedRepoList
│       └── RepoCard (repo_url, chunk_count, last_indexed)
└── ChatWindow
    ├── ModelSelector (Llama3 / CodeLlama)
    ├── MessageThread
    │   ├── UserMessage
    │   └── AssistantMessage
    │       ├── StreamingText
    │       └── SourceCards
    │           └── CodeSnippetCard (file_path, lines, collapsible)
    └── QueryInput
```

### Key Frontend Code Patterns

```javascript
// src/hooks/useStreamingChat.js

import { useRef, useState } from "react";

export function useStreamingChat() {
  const [messages, setMessages] = useState([]);
  const wsRef = useRef(null);

  const sendMessage = (repoId, question, model = "codellama:13b") => {
    wsRef.current = new WebSocket("ws://localhost:8000/api/v1/ws/chat");

    let assistantMessage = { role: "assistant", content: "", sources: [] };

    wsRef.current.onopen = () => {
      wsRef.current.send(JSON.stringify({ repo_id: repoId, question, model }));
    };

    wsRef.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "token") {
        assistantMessage.content += data.token;
        setMessages(prev => [...prev.slice(0, -1), { ...assistantMessage }]);
      } else if (data.type === "sources") {
        assistantMessage.sources = data.sources;
        setMessages(prev => [...prev.slice(0, -1), { ...assistantMessage }]);
      } else if (data.type === "done") {
        wsRef.current.close();
      }
    };

    setMessages(prev => [
      ...prev,
      { role: "user", content: question },
      { ...assistantMessage }
    ]);
  };

  return { messages, sendMessage };
}
```

---

## 10. Docker Compose Stack

```yaml
# docker-compose.yml

version: "3.9"

services:

  ollama:
    image: ollama/ollama:latest
    container_name: oracle-ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia          # remove if no GPU
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 15s
      timeout: 10s
      retries: 5

  model-puller:
    image: ollama/ollama:latest
    container_name: oracle-model-puller
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: >
      sh -c "
        OLLAMA_HOST=http://ollama:11434 ollama pull nomic-embed-text &&
        OLLAMA_HOST=http://ollama:11434 ollama pull codellama:13b &&
        OLLAMA_HOST=http://ollama:11434 ollama pull llama3:8b
      "
    volumes:
      - ./ollama_models:/root/.ollama
    restart: "no"

  chroma:
    image: chromadb/chroma:latest
    container_name: oracle-chroma
    ports:
      - "8001:8000"
    volumes:
      - ./chroma_data:/chroma_db
    environment:
      - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["http://localhost:5173","http://localhost:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: oracle-backend
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma_db
      - ./repos_cache:/tmp/repos
    environment:
      - OLLAMA_URL=http://ollama:11434
      - CHROMA_HOST=http://chroma:8000
      - LOG_LEVEL=info
    depends_on:
      ollama:
        condition: service_healthy
      chroma:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 15s
      timeout: 10s
      retries: 5

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: oracle-frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE=http://localhost:8000
      - VITE_WS_BASE=ws://localhost:8000
    depends_on:
      backend:
        condition: service_healthy

volumes:
  ollama_models:
  chroma_data:
  repos_cache:
```

### Backend Dockerfile

```dockerfile
# backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Backend requirements.txt

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

---

## 11. Security & Data Sovereignty Checklist

| Concern | Mitigation |
|---|---|
| Code never leaves machine | All inference via local Ollama; no OpenAI/Anthropic API calls |
| GitHub tokens | Pass via `.env`, never hardcode; mount as Docker secret |
| Private repos | Use `gitpython` with SSH key or personal access token from env |
| ChromaDB data | Mounted local volume; never exposed to internet |
| CORS | Restricted to `localhost:5173` only |
| Rate limiting | Add `slowapi` to FastAPI for ingestion endpoint (prevent re-index spam) |
| Input validation | Pydantic schemas on all endpoints; reject non-GitHub URLs |
| Repo size guard | Check repo size before cloning; reject > 500MB |

```python
# .env (never commit this)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
OLLAMA_URL=http://ollama:11434
CHROMA_HOST=http://chroma:8000
MAX_REPO_SIZE_MB=500
```

---

## 12. Testing & Evaluation Framework

### Retrieval Quality Metrics

| Metric | Tool | Target |
|---|---|---|
| Recall@5 | Custom eval script | > 0.70 |
| MRR (Mean Reciprocal Rank) | Custom eval script | > 0.60 |
| Answer faithfulness | LLM-as-Judge | > 0.80 |
| p95 query latency | Locust / manual timing | < 8 seconds (GPU) |

### Golden QA Test Set (Build This First)

For any repo you index, manually create 20–30 question/answer pairs:

```json
[
  {
    "question": "Where is MongoDB initialized and what is the database name?",
    "expected_file": "src/db/connection.py",
    "expected_symbol": "init_db",
    "answer_keywords": ["MongoClient", "DATABASE_URL", "db_name"]
  },
  {
    "question": "How does the WebSocket handler authenticate incoming connections?",
    "expected_file": "src/ws/handler.py",
    "expected_symbol": "on_connect",
    "answer_keywords": ["token", "verify_jwt", "401"]
  }
]
```

### Eval Script

```python
# tests/eval_retrieval.py

def evaluate_retrieval(test_set, retriever, k=5):
    hits = 0
    mrr_sum = 0
    for item in test_set:
        chunks = retriever.retrieve(item["question"], final_k=k)
        files = [c["file_path"] for c in chunks]
        symbols = [c["symbol_name"] for c in chunks]
        hit = item["expected_file"] in files or item["expected_symbol"] in symbols
        if hit:
            rank = next((i+1 for i, c in enumerate(chunks)
                         if c["symbol_name"] == item["expected_symbol"]), k+1)
            mrr_sum += 1 / rank
            hits += 1
    recall = hits / len(test_set)
    mrr = mrr_sum / len(test_set)
    print(f"Recall@{k}: {recall:.3f}  MRR: {mrr:.3f}")
    return recall, mrr
```

---

## 13. Phased Roadmap (Weeks)

```
Week 1  ──────────────────────────────────────────────────────────
  Day 1:  Environment setup, Ollama + model pulls, Docker test
  Day 2:  tree-sitter grammar install, ASTChunker for Python/JS
  Day 3:  EmbeddingService + ChromaWriter + basic ingestion test
  Day 4:  GitCloner + IngestionOrchestrator, run on a real repo
  Day 5:  BM25 index builder + DenseRetriever + RRF fusion

Week 2  ──────────────────────────────────────────────────────────
  Day 6:  Cross-encoder reranker integration + latency profiling
  Day 7:  PromptBuilder + OllamaClient streaming + guardrail
  Day 8:  FastAPI wiring: all routes, Pydantic schemas, CORS
  Day 9:  WebSocket streaming endpoint + SSE ingestion progress
  Day 10: Full end-to-end test: clone → ingest → query → answer

Week 3  ──────────────────────────────────────────────────────────
  Day 11: React scaffold, RepoManager + IngestButton
  Day 12: ChatWindow + useStreamingChat hook
  Day 13: SourceCards (collapsible code snippets with provenance)
  Day 14: Docker Compose full stack, model-puller service
  Day 15: Golden QA eval script, fix top retrieval failures

Week 4  ──────────────────────────────────────────────────────────
  Stretch: Multi-repo support (query across multiple collections)
  Stretch: Dependency graph (tree-sitter → call graph → KG retrieval)
  Stretch: File watcher for hot re-indexing on local repo changes
  Stretch: Model benchmarking UI (Llama3 vs CodeLlama side-by-side)
```

---

## 14. File System Architecture

This section documents every directory and file in the project — what it is, what it owns, and why it exists. Use this as your ground truth when navigating the codebase.

### Top-Level Project Tree

```
codebase-oracle/
│
├── docker-compose.yml          ← Orchestrates all 5 services (single source of truth)
├── .env                        ← Secrets & env vars (NEVER commit to git)
├── .gitignore
├── README.md
│
├── backend/                    ← Python FastAPI application
├── frontend/                   ← React + Vite SPA
├── chroma_data/                ← ChromaDB persistent storage (Docker volume mount)
├── ollama_models/              ← Ollama model weights cache (Docker volume mount)
├── repos_cache/                ← Cloned GitHub repos (ephemeral, can be wiped safely)
└── tests/                      ← Evaluation scripts & golden QA sets
```

---

### `/backend/` — FastAPI Application

```
backend/
│
├── Dockerfile                  ← Python 3.11-slim image, installs requirements, runs uvicorn
├── requirements.txt            ← All Python dependencies pinned to exact versions
├── main.py                     ← FastAPI app factory: mounts routers, CORS, lifespan events
│
├── api/                        ← HTTP route handlers (thin layer — no business logic here)
│   ├── __init__.py
│   ├── routes_ingest.py        ← POST /ingest, GET /ingest/{job_id}/status (SSE)
│   ├── routes_query.py         ← POST /query, WS /ws/chat (streaming)
│   └── routes_repos.py         ← GET /repos, DELETE /repos/{repo_id}, GET /health
│
├── ingestion/                  ← Everything about turning a repo into indexed chunks
│   ├── __init__.py
│   ├── orchestrator.py         ← Top-level ingestion coordinator; calls all sub-modules
│   ├── git_cloner.py           ← Clone/pull GitHub repos via gitpython; size guard
│   ├── file_walker.py          ← Recursively walk repo tree; map extensions → languages
│   ├── ast_chunker.py          ← tree-sitter AST parser; recursive split-merge algorithm
│   ├── embedding_service.py    ← Batch-embed chunks via Ollama nomic-embed-text
│   ├── chroma_writer.py        ← Upsert chunks + embeddings + metadata into ChromaDB
│   └── bm25_builder.py         ← Build & persist BM25 index from a ChromaDB collection
│
├── retrieval/                  ← Everything about answering a query
│   ├── __init__.py
│   ├── hybrid_retriever.py     ← BM25 + dense search → RRF fusion → cross-encoder rerank
│   ├── dense_retriever.py      ← ChromaDB cosine similarity search wrapper
│   ├── sparse_retriever.py     ← BM25Okapi wrapper; loads index from disk
│   ├── rrf_fusion.py           ← Reciprocal Rank Fusion merge logic
│   ├── reranker.py             ← cross-encoder/ms-marco-MiniLM-L6-v2 wrapper
│   ├── prompt_builder.py       ← Assembles system prompt + code context + user query
│   └── ollama_client.py        ← Async streaming POST to Ollama /api/chat
│
├── models/                     ← Pydantic schemas (shared across api/ and services)
│   ├── __init__.py
│   └── schemas.py              ← IngestRequest, QueryRequest, QueryResponse, SourceChunk, etc.
│
├── core/                       ← App-wide singletons and config
│   ├── __init__.py
│   ├── config.py               ← Pydantic Settings: reads from .env (OLLAMA_URL, CHROMA_HOST…)
│   ├── dependencies.py         ← FastAPI dependency injectors (get_chroma_client, get_embedder…)
│   └── logger.py               ← Structured JSON logging config (uvicorn + app logs unified)
│
└── jobs/                       ← Background task management
    ├── __init__.py
    └── job_store.py            ← In-memory dict {job_id → {status, progress, error}}
                                   (swap for Redis if you need persistence across restarts)
```

#### File Ownership Rules

| Directory | Owns | Does NOT own |
|---|---|---|
| `api/` | HTTP request/response shape, status codes | Business logic, DB calls |
| `ingestion/` | Chunking, embedding, writing to DB | Query-time logic |
| `retrieval/` | Search, ranking, prompt assembly, LLM call | Ingestion, DB writes |
| `models/` | Shared data contracts (Pydantic) | Any logic |
| `core/` | App-wide config, DI, logging | Feature logic |
| `jobs/` | Job lifecycle tracking | Actual job execution |

---

### `/frontend/` — React + Vite SPA

```
frontend/
│
├── Dockerfile                  ← Node 20-alpine; npm install; vite dev server on :5173
├── package.json
├── vite.config.js              ← Proxy /api → backend:8000 for dev; WS passthrough
├── index.html                  ← Single HTML entry point
│
├── public/
│   └── favicon.ico
│
└── src/
    ├── main.jsx                ← React DOM root mount
    ├── App.jsx                 ← Top-level layout: Sidebar + ChatWindow
    │
    ├── components/
    │   ├── Sidebar/
    │   │   ├── RepoManager.jsx         ← URL input, branch picker, ingest trigger
    │   │   ├── IngestionProgress.jsx   ← SSE-driven progress bar + file counter
    │   │   └── IndexedRepoList.jsx     ← Cards: repo name, chunk count, last indexed
    │   │
    │   ├── Chat/
    │   │   ├── ChatWindow.jsx          ← Scrollable message thread container
    │   │   ├── UserMessage.jsx         ← Right-aligned user bubble
    │   │   ├── AssistantMessage.jsx    ← Left-aligned; renders markdown + source cards
    │   │   ├── StreamingCursor.jsx     ← Blinking cursor during token stream
    │   │   └── QueryInput.jsx          ← Textarea + send button + model selector
    │   │
    │   └── Sources/
    │       ├── SourceCards.jsx         ← Horizontal scroll row of chunk cards
    │       └── CodeSnippetCard.jsx     ← Collapsible: file path, lines, syntax-highlighted code
    │
    ├── hooks/
    │   ├── useStreamingChat.js         ← WebSocket lifecycle; token accumulation; source parsing
    │   ├── useIngestion.js             ← SSE connection to /ingest/{job_id}/status
    │   └── useRepos.js                 ← GET /repos; delete repo action
    │
    ├── services/
    │   └── api.js                      ← Axios/fetch wrappers for all REST endpoints
    │
    └── styles/
        └── index.css                   ← Tailwind base (or plain CSS reset)
```

---

### `/chroma_data/` — Vector Database Persistent Storage

```
chroma_data/                    ← Mounted to /chroma_db inside the chroma container
│
└── {uuid}/                     ← One directory per ChromaDB collection
    ├── data_level0.bin         ← HNSW graph (the actual vector index)
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

> **Rule:** Never manually edit files here. Always interact through ChromaDB's Python client or HTTP API. Safe to delete a collection folder to wipe a repo's index — then re-ingest.

---

### `/ollama_models/` — LLM Weight Cache

```
ollama_models/                  ← Mounted to /root/.ollama inside the ollama container
│
└── models/
    ├── blobs/                  ← Raw model weight files (GGUF shards, multi-GB each)
    │   ├── sha256-{hash}       ← codellama:13b shards
    │   ├── sha256-{hash}       ← llama3:8b shards
    │   └── sha256-{hash}       ← nomic-embed-text shards
    └── manifests/
        └── registry.ollama.ai/
            ├── library/
            │   ├── codellama/  ← version manifests for codellama:13b
            │   └── llama3/     ← version manifests for llama3:8b
            └── nomic-ai/
                └── nomic-embed-text/
```

> **Rule:** This directory is the reason you never lose model weights between `docker compose down` and `docker compose up`. Without this volume mount, every restart re-downloads 8–26GB of weights.

---

### `/repos_cache/` — Cloned Repository Working Directory

```
repos_cache/                    ← Mounted to /tmp/repos inside the backend container
│
├── {md5_hash_of_url}/          ← One directory per repo (hash of repo URL)
│   ├── .git/                   ← Git metadata (depth=1 shallow clone)
│   ├── src/                    ← Actual source tree
│   └── ...
└── {another_repo_hash}/
```

> **Rule:** Treat as ephemeral. If you `docker compose down -v`, this is safe to wipe — re-ingesting will re-clone. Never store anything here that isn't reconstructable from the original GitHub repo.

---

### `/tests/` — Evaluation & Quality Assurance

```
tests/
│
├── golden_qa/
│   ├── {repo_name}_qa.json     ← Hand-crafted question/answer/expected_file test sets
│   └── README.md               ← How to add new test cases
│
├── eval_retrieval.py           ← Recall@K and MRR scoring against golden QA sets
├── eval_generation.py          ← LLM-as-Judge faithfulness scoring
├── load_test.py                ← Locust script: concurrent query simulation
└── conftest.py                 ← Pytest fixtures (spin up test ChromaDB collection)
```

---

### Root-Level Config Files

```
.env                            ← Runtime secrets (read by backend/core/config.py)
│                                  GITHUB_TOKEN, OLLAMA_URL, CHROMA_HOST,
│                                  DEFAULT_LLM_MODEL, MAX_REPO_SIZE_MB,
│                                  LOG_LEVEL, BM25_CACHE_DIR
│
.gitignore                      ← Must include: .env, chroma_data/, ollama_models/,
│                                  repos_cache/, __pycache__/, *.pyc, node_modules/
│
docker-compose.yml              ← Single source of truth for service wiring
│                                  (see Section 10 for full contents)
│
README.md                       ← Quick start: prerequisites, docker compose up,
                                   first ingest + query walkthrough
```

---

### Data Flow Mapped to File System

```
1. User submits repo URL
        │
        ▼
   api/routes_ingest.py          ← validates IngestRequest schema
        │
        ▼
   ingestion/orchestrator.py     ← coordinates the pipeline
        │
        ├──► ingestion/git_cloner.py      → writes to  repos_cache/{hash}/
        ├──► ingestion/file_walker.py     → reads from repos_cache/{hash}/
        ├──► ingestion/ast_chunker.py     → in-memory chunk dicts
        ├──► ingestion/embedding_service.py → HTTP → ollama:11434
        ├──► ingestion/chroma_writer.py   → writes to  chroma_data/{collection}/
        └──► ingestion/bm25_builder.py    → writes to  /tmp/bm25/{repo_id}.pkl

2. User sends a query
        │
        ▼
   api/routes_query.py           ← validates QueryRequest schema
        │
        ▼
   retrieval/hybrid_retriever.py
        │
        ├──► retrieval/dense_retriever.py   → reads from chroma_data/ (HTTP to chroma:8000)
        ├──► retrieval/sparse_retriever.py  → reads from /tmp/bm25/{repo_id}.pkl
        ├──► retrieval/rrf_fusion.py        → in-memory merge
        ├──► retrieval/reranker.py          → in-memory cross-encoder (CPU)
        └──► retrieval/prompt_builder.py    → assembles context string
                │
                ▼
        retrieval/ollama_client.py  → streams tokens → HTTP → ollama:11434
                │
                ▼
        api/routes_query.py  → streams tokens → WebSocket → frontend
```

---

## Appendix A — Key Research Citations

| Finding | Source |
|---|---|
| AST-based chunking: 70.1% Recall@5 vs 42.4% fixed-size | cAST paper, arxiv 2506.15655, June 2025 |
| Optimal chunk size: 400–512 tokens; quality drops above 2500 | hermes-agent issue #844, March 2026 |
| Hybrid retrieval recall: ~0.91 vs 0.72 (BM25 alone) | Medium: BM25 vs Hybrid RAG, March 2026 |
| Optimal hybrid weights: 30% sparse / 70% dense | arxiv 2511.10297, November 2025 |
| Cross-encoder reranking: +35–40% answer accuracy | ailog.fr reranking guide, February 2025 |
| Retrieve 50–100 candidates, rerank to top 8–10 | ailog.fr, Towards Data Science, 2025–2026 |
| Structural retrieval complements embedding retrieval for code | Codebase-Memory paper, arxiv 2603.27277, March 2026 |
| Never use single Ollama instance for both embed + generate | markaicode.com hybrid retrieval, May 2026 |

---

*Blueprint version 1.1 — June 2026 (added File System Architecture)*  
*Estimated total build time: 3–4 weeks solo (no prior RAG experience needed beyond NextWork baseline)*
