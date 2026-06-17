"""BM25 index builder with persistent JSON cache.

Cache strategy
--------------
The BM25 index (``rank_bm25.BM25Okapi``) is not serialisable, so we cache
the tokenized corpus instead:

    {BM25_CACHE_DIR}/{collection_name}.json

Each entry in the JSON file is:
    {"text": str, "tokens": list[str], <metadata fields>}

On load we reconstruct ``BM25Okapi(tokenized_corpus)`` in-memory.  This
keeps the cache file human-inspectable and avoids pickle.

The cache directory is a Docker-mounted volume (``/bm25_cache``) so that
indexes survive container restarts.  Using ``/tmp/bm25`` (ephemeral) caused
a full 30–60 s rebuild on every restart — hence this fix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class BM25Builder:
    """Builds a BM25 index over a ChromaDB collection with file-based caching."""

    def build_index(
        self,
        collection_name: str,
        chroma_client: chromadb.PersistentClient,
    ) -> tuple[BM25Okapi, list[dict]]:
        """Return ``(BM25Okapi index, corpus)`` for *collection_name*.

        Loads from ``{BM25_CACHE_DIR}/{collection_name}.json`` if present;
        otherwise fetches documents from ChromaDB, builds the index, and
        writes the cache file.
        """
        cache_path = Path(settings.BM25_CACHE_DIR) / f"{collection_name}.json"

        if cache_path.is_file():
            logger.info("Loading BM25 index from cache: %s", cache_path)
            return self._load_from_cache(cache_path)

        logger.info(
            "BM25 cache miss — building index for '%s'", collection_name
        )
        corpus = self._fetch_corpus(collection_name, chroma_client)
        index = self._build_and_persist(corpus, cache_path)
        return index, corpus

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_corpus(
        collection_name: str,
        chroma_client: chromadb.PersistentClient,
    ) -> list[dict]:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(include=["documents", "metadatas"])

        corpus: list[dict] = []
        for text, meta in zip(
            results.get("documents") or [],
            results.get("metadatas") or [],
        ):
            entry: dict[str, Any] = {"text": text}
            if meta:
                entry.update(meta)
            corpus.append(entry)

        logger.info("Fetched %d documents from '%s'", len(corpus), collection_name)
        return corpus

    @staticmethod
    def _build_and_persist(
        corpus: list[dict], cache_path: Path
    ) -> BM25Okapi:
        tokenized = [doc["text"].lower().split() for doc in corpus]
        index = BM25Okapi(tokenized)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Store tokens alongside corpus fields so we can rebuild without
        # re-tokenising.
        cache_entries = [
            {
                "tokens": tokens,
                **{k: v for k, v in doc.items()},
            }
            for doc, tokens in zip(corpus, tokenized)
        ]
        cache_path.write_text(
            json.dumps(cache_entries, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "BM25 cache written: %s (%d docs)", cache_path, len(corpus)
        )
        return index

    @staticmethod
    def _load_from_cache(
        cache_path: Path,
    ) -> tuple[BM25Okapi, list[dict]]:
        raw: list[dict] = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
        tokenized = [entry["tokens"] for entry in raw]
        # Reconstruct corpus without the "tokens" key
        corpus = [
            {k: v for k, v in entry.items() if k != "tokens"}
            for entry in raw
        ]
        index = BM25Okapi(tokenized)
        logger.info("BM25 index loaded from cache: %d docs", len(corpus))
        return index, corpus
