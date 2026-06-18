"""Dense retrieval via ChromaDB cosine-similarity query.

Design notes
------------
- Wraps the ChromaDB collection.query() call and normalises the result
  into the canonical chunk dict schema used across the pipeline.
- ChromaDB 0.5.x returns nested lists (one inner list per query).  Since
  we always query with a single embedding, we unpack index [0].
- ``score = 1 - distance`` converts cosine distance (0 = identical,
  2 = opposite) to a similarity score in [−1, 1].  Practically all
  real chunks score in [0, 1].
"""
from __future__ import annotations

import chromadb

from core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_K = 50


class DenseRetriever:
    """Retrieves top-k chunks from ChromaDB using a pre-computed query embedding."""

    def retrieve(
        self,
        query_embedding: list[float],
        collection: chromadb.Collection,
        k: int = _DEFAULT_K,
    ) -> list[dict]:
        """Return up to *k* chunks ranked by cosine similarity.

        Args:
            query_embedding: 768-dim vector produced by ``EmbeddingService``.
            collection: ChromaDB collection (cosine space, created by
                ``ChromaWriter``).
            k: Maximum number of results to return.

        Returns:
            List of chunk dicts, each containing all stored metadata keys
            plus ``text`` and ``score`` (similarity, higher is better).
        """
        count = collection.count()
        if count == 0:
            logger.debug("Dense retrieval skipped — empty collection")
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents: list[str] = result["documents"][0]
        metadatas: list[dict] = result["metadatas"][0]
        distances: list[float] = result["distances"][0]

        chunks: list[dict] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            chunk: dict = {
                "text": text,
                "score": 1.0 - dist,
                **meta,
            }
            chunks.append(chunk)

        logger.debug("Dense retrieval returned %d chunks (k=%d)", len(chunks), k)
        return chunks
