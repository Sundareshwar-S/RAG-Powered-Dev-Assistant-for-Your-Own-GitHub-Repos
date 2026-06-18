"""Sparse retrieval via BM25 keyword matching.

Design notes
------------
- Wraps ``rank_bm25.BM25Okapi.get_scores()`` and returns the top-k
  results as (score, chunk) pairs, preserving the same interface
  expected by ``RRFFusion``.
- Tokenisation is kept deliberately simple (lowercase + whitespace
  split) to match the tokenisation used in ``BM25Builder`` so that
  score comparisons are meaningful.
- ``numpy`` argsort is used for efficiency on large corpora; the result
  is reversed so that higher scores come first.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_K = 50


class SparseRetriever:
    """Retrieves top-k chunks from a BM25 index by keyword relevance."""

    def retrieve(
        self,
        query: str,
        bm25_index: BM25Okapi,
        corpus: list[dict],
        k: int = _DEFAULT_K,
    ) -> list[tuple[float, dict]]:
        """Return up to *k* (score, chunk) pairs ranked by BM25 score.

        Args:
            query: Raw query string from the user.
            bm25_index: Pre-built ``BM25Okapi`` instance (from
                ``BM25Builder``).
            corpus: Ordered list of chunk dicts that matches the corpus
                used to build *bm25_index*.
            k: Maximum number of results to return.

        Returns:
            List of ``(score, chunk_dict)`` tuples, sorted by score
            descending.  Chunks with zero score are excluded.
        """
        tokens = query.lower().split()
        scores: np.ndarray = bm25_index.get_scores(tokens)

        # argsort ascending → reverse for descending order
        ranked_indices = np.argsort(scores)[::-1]

        results: list[tuple[float, dict]] = []
        for idx in ranked_indices[:k]:
            score = float(scores[idx])
            if score <= 0.0:
                break
            results.append((score, corpus[idx]))

        logger.debug("Sparse retrieval returned %d chunks (k=%d)", len(results), k)
        return results
