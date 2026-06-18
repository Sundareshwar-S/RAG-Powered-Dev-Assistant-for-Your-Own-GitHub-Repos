"""Retrieval quality evaluation for CodeBase Oracle.

Metrics
-------
- Recall@K:  fraction of questions where the expected file OR symbol appears
             in the top-K retrieved chunks.
- MRR:       Mean Reciprocal Rank — average of 1/rank for the first hit.

Targets (Phase 7 gate)
-----------------------
- Recall@5 > 0.70
- MRR      > 0.60

Usage
-----
# Against a running stack with markupsafe already indexed:
    python tests/eval_retrieval.py

# Custom QA file and repo:
    python tests/eval_retrieval.py \
        --qa tests/golden_qa/markupsafe_qa.json \
        --repo-id <8-char-id> \
        --k 5

# Offline integration tests (no running stack):
    pytest tests/test_phase1_integration.py tests/test_phase2_retrieval.py -v
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make backend modules importable when run from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

_RECALL_TARGET = 0.70
_MRR_TARGET = 0.60


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def _hit(
    chunks: list[dict[str, Any]],
    expected_file: str,
    expected_symbol: str,
    match_mode: str = "file_or_symbol",
) -> int | None:
    """Return the 1-based rank of the first matching chunk, or None.

    Args:
        match_mode: One of:
            ``"file_or_symbol"`` (default) — hit if either expected_file or
                expected_symbol matches.  Used for most questions.
            ``"file"`` — hit only if expected_file appears in file_path.
                Use for questions about class methods that the AST chunker
                merges into a single parent-class chunk.
            ``"symbol"`` — hit only if expected_symbol matches symbol_name
                exactly.  Use for top-level functions with unambiguous names.
    """
    for rank, chunk in enumerate(chunks, start=1):
        file_path: str = chunk.get("file_path", "")
        symbol_name: str = chunk.get("symbol_name", "")

        file_match = bool(expected_file and expected_file in file_path)
        symbol_match = bool(expected_symbol and expected_symbol == symbol_name)

        if match_mode == "file":
            if file_match:
                return rank
        elif match_mode == "symbol":
            if symbol_match:
                return rank
        else:  # "file_or_symbol"
            if file_match or symbol_match:
                return rank
    return None


def compute_recall_at_k(
    results: list[int | None],
    k: int,
) -> float:
    """Fraction of queries with a hit in the top-k results."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r is not None and r <= k)
    return hits / len(results)


def compute_mrr(results: list[int | None]) -> float:
    """Mean Reciprocal Rank; queries with no hit contribute 0."""
    if not results:
        return 0.0
    return sum(1 / r for r in results if r is not None) / len(results)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


async def evaluate_retrieval(
    test_set: list[dict[str, Any]],
    retriever: Any,  # HybridRetriever — typed loosely to avoid import issues in mocks
    k: int = 5,
) -> dict[str, float]:
    """Evaluate retrieval quality over *test_set*.

    Args:
        test_set:  List of QA dicts with ``question``, ``expected_file``,
                   ``expected_symbol``, and ``answer_keywords`` keys.
        retriever: An object with an async ``retrieve(query: str) -> list[dict]``
                   method (HybridRetriever or compatible mock).
        k:         Recall@K cut-off.

    Returns:
        Dict with ``recall_at_k`` and ``mrr`` as floats.
    """
    ranks: list[int | None] = []

    print(f"\n{'='*70}")
    print(f"  Retrieval Evaluation — Recall@{k} and MRR")
    print(f"  Questions: {len(test_set)}   k={k}")
    print(f"{'='*70}\n")

    for i, entry in enumerate(test_set, start=1):
        question: str = entry["question"]
        expected_file: str = entry.get("expected_file", "")
        expected_symbol: str = entry.get("expected_symbol", "")
        match_mode: str = entry.get("match_mode", "file_or_symbol")

        try:
            chunks = await retriever.retrieve(question, final_k=k)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:02d}] ERROR retrieving: {exc}")
            ranks.append(None)
            continue

        rank = _hit(chunks, expected_file, expected_symbol, match_mode=match_mode)
        ranks.append(rank)

        status = f"PASS (rank={rank})" if rank is not None and rank <= k else "FAIL"
        symbol_found = next(
            (c.get("symbol_name", "") for c in chunks if c.get("symbol_name") == expected_symbol),
            "-",
        )
        print(
            f"  [{i:02d}] {status:<20} | mode={match_mode:<15} "
            f"| expected={expected_symbol!r:<25} | found={symbol_found!r:<20} "
            f"| q={question[:45]!r}"
        )

    recall = compute_recall_at_k(ranks, k)
    mrr = compute_mrr(ranks)

    print(f"\n{'='*70}")
    print(f"  Recall@{k} = {recall:.3f}  (target > {_RECALL_TARGET})")
    print(f"  MRR      = {mrr:.3f}  (target > {_MRR_TARGET})")
    print(f"  {'PASS' if recall > _RECALL_TARGET and mrr > _MRR_TARGET else 'NEEDS TUNING'}")
    print(f"{'='*70}\n")

    return {"recall_at_k": recall, "mrr": mrr, "k": k, "n": len(test_set)}


# ---------------------------------------------------------------------------
# CLI entry point — requires a running stack
# ---------------------------------------------------------------------------


async def _main(qa_path: Path, repo_id: str, k: int) -> None:
    from core.dependencies import get_bm25_data, get_chroma_client, get_embed_service
    from retrieval.hybrid_retriever import HybridRetriever

    test_set: list[dict] = json.loads(qa_path.read_text())

    chroma_client = get_chroma_client()
    embed_service = get_embed_service()
    collection_name = f"repo_{repo_id}"

    bm25_index, corpus = get_bm25_data(collection_name, chroma_client)
    retriever = HybridRetriever(
        collection_name=collection_name,
        chroma_client=chroma_client,
        embed_service=embed_service,
        bm25_index=bm25_index,
        corpus=corpus,
    )

    metrics = await evaluate_retrieval(test_set, retriever, k=k)
    sys.exit(0 if metrics["recall_at_k"] > _RECALL_TARGET and metrics["mrr"] > _MRR_TARGET else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path(__file__).parent / "golden_qa" / "markupsafe_qa.json",
        help="Path to golden QA JSON file.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="8-char repo_id (md5 prefix). If omitted, derived from markupsafe URL.",
    )
    parser.add_argument("--k", type=int, default=5, help="Recall@K cut-off (default 5).")
    args = parser.parse_args()

    if args.repo_id is None:
        import hashlib
        args.repo_id = hashlib.md5(
            b"https://github.com/pallets/markupsafe", usedforsecurity=False
        ).hexdigest()[:8]
        print(f"[info] Using derived repo_id={args.repo_id!r} for markupsafe")

    asyncio.run(_main(args.qa, args.repo_id, args.k))
