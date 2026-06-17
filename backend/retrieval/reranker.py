"""Cross-encoder reranker for final passage scoring.

Model choice
------------
``cross-encoder/ms-marco-MiniLM-L-6-v2`` (6-layer MiniLM) is the standard
lightweight reranker for passage retrieval:
- ~22 M parameters, ~85 MB on disk
- MS MARCO MRR@10 ≈ 0.390 (competitive with much larger models)
- Inference on CPU: ~50 ms / 8 pairs on a modern laptop

Score normalisation
-------------------
``apply_softmax=True`` is passed to ``predict()`` so that raw logits (which
for code queries typically range from −12 to −4 for this model) are converted
to sigmoid probabilities in [0, 1].  Without this, the ``has_sufficient_context``
guardrail in ``prompt_builder`` would compare 0.25 against a logit like −8 and
always return False, silently preventing any LLM call.

Text pre-clipping
-----------------
``_MAX_LENGTH=512`` tokens is the CrossEncoder's hard limit.  Code chunks
stored by nomic-embed-text can be much longer (up to 8192 tokens).  Feeding
an unclipped chunk causes silent truncation mid-symbol, which degrades the
relevance score of long functions.  We pre-clip each chunk to
``_RERANKER_MAX_CHARS`` before building pairs, which approximates the token
limit conservatively.

Lazy loading
------------
The CrossEncoder is instantiated on the first call to ``rerank()``.  A
``threading.Lock`` prevents concurrent download on simultaneous requests.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from core.debug_log import agent_debug_log
from core.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_LENGTH = 512
# Conservative char limit ≈ 480 tokens × 4 chars/token.  Pre-clipping avoids
# the CrossEncoder silently scoring an incomplete, truncated code fragment.
_RERANKER_MAX_CHARS = 1920
_DEFAULT_TOP_K = 8

# Module-level singleton; None until first rerank() call.
# Protected by a lock to prevent concurrent download on simultaneous requests.
_cross_encoder: CrossEncoder | None = None
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:  # double-checked locking
                from sentence_transformers import CrossEncoder  # lazy import

                logger.info("Loading CrossEncoder model: %s", _MODEL_NAME)
                _cross_encoder = CrossEncoder(_MODEL_NAME, max_length=_MAX_LENGTH)
    return _cross_encoder


class Reranker:
    """Reranks candidate chunks using a cross-encoder relevance model."""

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[dict]:
        """Score *candidates* against *query* and return the top-*top_k* results.

        Args:
            query: User question string.
            candidates: Chunk dicts (e.g. from ``RRFFusion``).  Each must
                contain a ``text`` key.
            top_k: How many chunks to return after reranking.

        Returns:
            Up to *top_k* chunk dicts sorted by cross-encoder score
            descending.  Each dict gains a ``score`` key with a sigmoid
            probability in [0, 1] (higher = more relevant).  The probability
            is used by ``has_sufficient_context`` to decide whether to call
            the LLM.
        """
        if not candidates:
            return []

        cross_encoder = _get_cross_encoder()
        # Pre-clip text to avoid silent truncation of long code chunks.
        pairs = [
            (query, chunk["text"][:_RERANKER_MAX_CHARS]) for chunk in candidates
        ]
        # sentence-transformers 3.x ignores apply_softmax for ms-marco models and
        # returns raw logits.  Normalise with softmax across this query's
        # candidates so scores land in [0, 1] and the guardrail threshold works.
        logits: np.ndarray = cross_encoder.predict(pairs, apply_softmax=False)
        if len(logits) == 1:
            probabilities = np.array([1.0], dtype=float)
        else:
            shifted = logits - logits.max()
            exp_logits = np.exp(shifted)
            probabilities = exp_logits / exp_logits.sum()
        scores: list[float] = probabilities.tolist()

        agent_debug_log(
            "reranker.py:rerank",
            "Reranker score normalisation",
            {
                "query_prefix": query[:80],
                "candidate_count": len(candidates),
                "raw_logits_sample": logits[:3].tolist(),
                "max_logit": float(logits.max()) if len(logits) else None,
                "max_probability": float(probabilities.max()) if len(probabilities) else None,
            },
            hypothesis_id="H2",
        )

        scored = [
            {**chunk, "score": score}
            for chunk, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda c: c["score"], reverse=True)

        top = scored[:top_k]
        logger.debug(
            "Reranker: %d candidates → top-%d (best score=%.4f)",
            len(candidates),
            top_k,
            top[0]["score"] if top else float("-inf"),
        )
        return top
