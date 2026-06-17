"""Pytest configuration and shared fixtures for the CodeBase Oracle test suite.

Import path
-----------
All backend modules live under ``backend/`` in the repository root.  By
inserting ``backend/`` at the front of ``sys.path`` we make imports like
``from core.config import settings`` work both in Docker (where the WORKDIR
is already ``/app == backend/``) and locally from the repo root.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import chromadb
import pytest

# Make all backend modules importable without the "backend." prefix
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

_GOLDEN_QA_DIR = Path(__file__).parent / "golden_qa"


# ---------------------------------------------------------------------------
# Event loop (needed for async tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop so async fixtures can be shared."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_chroma(tmp_path: Path) -> chromadb.PersistentClient:
    """Isolated ChromaDB client backed by a temporary directory."""
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


# ---------------------------------------------------------------------------
# Embedding service mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embed_service():
    """EmbeddingService stub that returns 768-dim zero vectors instantly.

    Use this in unit tests that don't need real embeddings so that tests
    pass without a running Ollama instance.
    """
    service = MagicMock()
    service.embed_batch = AsyncMock(
        side_effect=lambda texts: [[0.0] * 768 for _ in texts]
    )
    return service


# ---------------------------------------------------------------------------
# Sample Python source fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_python_source() -> str:
    return """
def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


class Calculator:
    def multiply(self, a: int, b: int) -> int:
        return a * b
"""


# ---------------------------------------------------------------------------
# Pre-indexed ChromaDB collection for retrieval unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def pre_indexed_collection(
    tmp_chroma: chromadb.PersistentClient,
    sample_python_source: str,
) -> chromadb.Collection:
    """A small ChromaDB collection pre-loaded from sample_python_source.

    Provides a realistic retrieval target for unit tests that want to call
    ``DenseRetriever.retrieve()`` or ``HybridRetriever`` without a full
    Ollama + real-embedding stack.  Embeddings are 768-dim zero vectors so
    cosine-similarity ranking is non-deterministic; these tests should assert
    only on structure (count, field presence), not on rank order.
    """
    from ingestion.ast_chunker import ASTChunker

    chunker = ASTChunker("python")
    chunks = chunker.chunk(sample_python_source, "tests/sample.py")

    collection = tmp_chroma.get_or_create_collection(
        "repo_test01",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = [[0.0] * 768 for _ in chunks]
    metadatas = [
        {
            "file_path": c.get("file_path", ""),
            "language": c.get("language", "python"),
            "chunk_type": c.get("chunk_type", "function"),
            "start_line": int(c.get("start_line", 0)),
            "end_line": int(c.get("end_line", 0)),
            "symbol_name": c.get("symbol_name", ""),
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


# ---------------------------------------------------------------------------
# Golden QA dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def golden_qa_set() -> list[dict]:
    """Load the markupsafe golden QA test set from disk.

    Used by parametrised pytest tests that wrap evaluate_retrieval() and
    evaluate_generation() to ensure the eval scripts are importable and
    their metric functions return values in the expected range.

    Skip automatically if the JSON file is not present (e.g. in a stripped
    CI environment).
    """
    qa_path = _GOLDEN_QA_DIR / "markupsafe_qa.json"
    if not qa_path.exists():
        pytest.skip(f"Golden QA file not found: {qa_path}")
    return json.loads(qa_path.read_text())


# ---------------------------------------------------------------------------
# Shared chunk helper + HybridRetriever factory
# ---------------------------------------------------------------------------


def make_chunk(text: str = "def foo(): pass", idx: int = 0, score: float = 0.5) -> dict:
    return {
        "text": text,
        "file_path": f"src/file{idx}.py",
        "language": "python",
        "chunk_type": "function_definition",
        "start_line": idx * 10 + 1,
        "end_line": idx * 10 + 5,
        "symbol_name": f"symbol_{idx}",
        "score": score,
    }


@pytest.fixture
def hybrid_retriever_factory(mock_embed_service, tmp_chroma):
    """Build a HybridRetriever wired to a real Chroma collection + BM25 corpus."""

    def _factory(collection: chromadb.Collection, corpus: list[dict] | None = None):
        from rank_bm25 import BM25Okapi

        from retrieval.hybrid_retriever import HybridRetriever

        if corpus is None:
            result = collection.get(include=["documents", "metadatas"])
            corpus = [
                {
                    "text": doc,
                    **(meta or {}),
                }
                for doc, meta in zip(result["documents"], result["metadatas"])
            ]

        tokenized = [c["text"].lower().split() for c in corpus]
        bm25_index = BM25Okapi(tokenized)

        return HybridRetriever(
            collection_name=collection.name,
            chroma_client=tmp_chroma,
            embed_service=mock_embed_service,
            bm25_index=bm25_index,
            corpus=corpus,
        )

    return _factory

