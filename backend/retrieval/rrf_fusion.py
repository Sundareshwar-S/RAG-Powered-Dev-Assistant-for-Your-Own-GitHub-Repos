"""Reciprocal Rank Fusion (RRF) for merging dense and sparse result lists.

Algorithm
---------
For every chunk appearing in either list, compute:

    rrf_score(chunk) = sum_over_lists( 1 / (rank_in_list + k) )

where ``rank`` is 1-based and ``k=60`` is the standard smoothing constant
(Cormack et al., 2009).

Deduplication key is ``chunk["text"][:100]`` — long enough to distinguish
different chunks but cheap to compute and robust to minor metadata
differences between the dense and sparse representations of the same doc.

The output list carries the ``rrf_score`` field (float) so that downstream
components (e.g. the reranker) can inspect fusion confidence.
"""
from __future__ import annotations

from core.logger import get_logger

logger = get_logger(__name__)

_RRF_K = 60
_TOP_N = 100


def merge(
    dense_chunks: list[dict],
    sparse_scored_chunks: list[tuple[float, dict]],
    k: int = _RRF_K,
) -> list[dict]:
    """Fuse dense and sparse ranked lists via RRF.

    Args:
        dense_chunks: Ordered list of chunk dicts from ``DenseRetriever``
            (index 0 = rank 1, highest similarity first).
        sparse_scored_chunks: Ordered list of ``(bm25_score, chunk_dict)``
            tuples from ``SparseRetriever`` (index 0 = rank 1).
        k: RRF smoothing constant.  Standard value is 60.

    Returns:
        Up to ``_TOP_N`` (100) chunk dicts sorted by RRF score descending.
        Each dict includes an ``rrf_score`` key with the accumulated score.
        Duplicate text chunks (same ``text[:100]``) are merged into a
        single entry using the higher-ranked source's metadata.
    """
    # key → (rrf_score_accumulator, representative_chunk_dict)
    fused: dict[str, list] = {}

    def _chunk_key(chunk: dict) -> str:
        """Stable dedup key: prefer structural identity over text prefix.

        ``text[:100]`` collapses empty-text chunks and any two chunks whose
        code starts identically (e.g. two functions both beginning with
        ``def __init__``).  Using ``file_path:start_line:end_line`` gives a
        precise identity based on source location.  Fall back to text prefix
        only when metadata is absent.
        """
        fp = chunk.get("file_path")
        sl = chunk.get("start_line")
        el = chunk.get("end_line")
        if fp and sl is not None and el is not None:
            return f"{fp}:{sl}:{el}"
        return chunk.get("text", "")[:100]

    def _add(rank: int, chunk: dict) -> None:
        key = _chunk_key(chunk)
        score = 1.0 / (rank + k)
        if key in fused:
            fused[key][0] += score
        else:
            fused[key] = [score, chunk]

    for rank, chunk in enumerate(dense_chunks, start=1):
        _add(rank, chunk)

    for rank, (_, chunk) in enumerate(sparse_scored_chunks, start=1):
        _add(rank, chunk)

    # Build output list with rrf_score attached
    results: list[dict] = []
    for score, chunk in fused.values():
        entry = {**chunk, "rrf_score": score}
        results.append(entry)

    results.sort(key=lambda c: c["rrf_score"], reverse=True)
    top = results[:_TOP_N]

    logger.debug(
        "RRF fusion: %d dense + %d sparse → %d fused (returned %d)",
        len(dense_chunks),
        len(sparse_scored_chunks),
        len(fused),
        len(top),
    )
    return top
