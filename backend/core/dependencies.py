"""FastAPI dependency injection — process-scoped singletons.

All heavy objects (ChromaDB client, embedding service) are created once per
process via ``@lru_cache(maxsize=1)`` and reused across requests.  This avoids
the cost of reconnecting to ChromaDB or re-loading sentence-transformers on
every request.

BM25 data is loaded on demand per collection because it depends on the
collection name (which varies per repo).  Results are cached in the module-
level ``_bm25_cache`` dict so repeated queries against the same repo do not
trigger a rebuild.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from core.config import Settings, settings as _settings
from core.logger import get_logger
from ingestion.bm25_builder import BM25Builder
from ingestion.embedding_service import EmbeddingService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level BM25 cache — keyed by collection name
# ---------------------------------------------------------------------------
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}
_bm25_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB persistent client (created once)."""
    logger.info("Initialising ChromaDB PersistentClient at /chroma_db")
    return chromadb.PersistentClient(path="/chroma_db")


@lru_cache(maxsize=1)
def get_embed_service() -> EmbeddingService:
    """Return the shared EmbeddingService instance (created once)."""
    logger.info("Initialising EmbeddingService")
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton."""
    return _settings


def get_bm25_data(
    collection_name: str,
    chroma_client: chromadb.PersistentClient | None = None,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Return ``(BM25Okapi index, corpus)`` for *collection_name*.

    Loads from the file cache on first call; subsequent calls return the
    in-memory cached result without disk I/O.

    Args:
        collection_name: ChromaDB collection name, e.g. ``"repo_abc12345"``.
        chroma_client:  Client to use for a cache miss.  Falls back to the
            shared singleton when ``None``.

    Returns:
        A tuple of (BM25Okapi index, list of corpus dicts).

    Raises:
        Exception: Re-raises any error from :class:`BM25Builder`.
    """
    with _bm25_lock:
        if collection_name in _bm25_cache:
            return _bm25_cache[collection_name]

        # Build outside the lock to avoid blocking other cache reads, but
        # re-check inside the lock afterward to handle concurrent callers.
        # We hold the lock for the build here (simpler, correct) because
        # BM25 builds are infrequent and short relative to embedding calls.
        client = chroma_client or get_chroma_client()
        builder = BM25Builder()
        index, corpus = builder.build_index(collection_name, client)
        _bm25_cache[collection_name] = (index, corpus)

    return index, corpus


def invalidate_bm25_cache(collection_name: str) -> None:
    """Remove *collection_name* from the in-memory BM25 cache.

    Called after a repo is deleted so stale data is not served to subsequent
    requests.
    """
    with _bm25_lock:
        _bm25_cache.pop(collection_name, None)
