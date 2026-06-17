"""Generation quality evaluation for CodeBase Oracle.

Metrics
-------
- keyword_hit_rate:  Fraction of ``answer_keywords`` that appear in the
                     generated answer (case-insensitive substring match).
                     Averaged across all QA entries.
- faithfulness:      LLM-as-Judge score.  A second Ollama call judges whether
                     each answer cites a plausible code path.  Returns a float
                     between 0 and 1 (fraction judged faithful).

Targets (Phase 7 gate)
-----------------------
- keyword_hit_rate > 0.80

Usage
-----
# Against a running stack with markupsafe already indexed:
    python tests/eval_generation.py

# Custom QA file, repo, and model:
    python tests/eval_generation.py \
        --qa tests/golden_qa/markupsafe_qa.json \
        --repo-id <8-char-id> \
        --model qwen2.5-coder:7b

# Pytest wrapper (mocked, no running stack required):
    pytest tests/ -k "eval_generation" -v
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

_KEYWORD_HIT_RATE_TARGET = 0.80


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _keyword_hit_rate_for_entry(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords present in answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


_JUDGE_SYSTEM = (
    "You are a strict code-QA judge. You will be shown a question, a generated answer, "
    "and retrieved source snippets. Reply with exactly one word: 'faithful' if the answer "
    "accurately reflects the source code and cites real symbols, file paths, or logic. "
    "Reply with 'unfaithful' if the answer hallucinated details not present in the sources."
)


def _build_judge_prompt(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build an LLM-as-Judge prompt.

    Uses up to 8 chunks so the judge sees the same context window that was
    presented to the generator — avoiding false-unfaithful verdicts when the
    answer cites lower-ranked chunks not visible to a 4-chunk truncation.
    """
    sources_block = "\n\n".join(
        f"[Chunk {i}] File: {c.get('file_path', '?')} Symbol: {c.get('symbol_name', '?')}\n"
        f"{c.get('text', '')[:400]}"
        for i, c in enumerate(chunks[:8], start=1)
    )
    user_content = (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Answer to judge:\n{answer}"
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


async def _judge_faithfulness(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    model: str,
    ollama_url: str,
) -> bool | None:
    """Ask the LLM to judge whether the answer is faithful to the sources.

    Returns:
        True  — judge replied "faithful"
        False — judge replied "unfaithful" (or any non-faithful response)
        None  — I/O error; caller should exclude from the rate denominator
    """
    import httpx

    messages = _build_judge_prompt(question, answer, chunks)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            verdict: str = resp.json()["message"]["content"].strip().lower()
            # Exact match to avoid "unfaithful" matching "faithful" via substring.
            return verdict == "faithful"
    except Exception:  # noqa: BLE001
        return None  # network error — caller excludes this from the rate denominator


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


async def evaluate_generation(
    test_set: list[dict[str, Any]],
    retriever: Any,
    stream_response_fn: Any,
    model: str = "qwen2.5-coder:7b",
    ollama_url: str = "http://localhost:11434",
    run_judge: bool = True,
) -> dict[str, float]:
    """Evaluate generation quality over *test_set*.

    Args:
        test_set:          List of QA dicts from a golden QA JSON file.
        retriever:         HybridRetriever (or compatible mock) with async ``retrieve``.
        stream_response_fn: Async generator function with signature
                            ``stream_response(messages, model) -> AsyncGenerator[str, None]``.
        model:             Ollama model name for generation.
        ollama_url:        Base URL for the LLM-as-Judge call.
        run_judge:         Set False to skip the faithfulness judge (faster, offline tests).

    Returns:
        Dict with ``keyword_hit_rate`` and ``faithfulness`` as floats.
    """
    from retrieval.prompt_builder import build_prompt, has_sufficient_context

    keyword_rates: list[float] = []
    faithfulness_verdicts: list[bool] = []

    print(f"\n{'='*70}")
    print(f"  Generation Evaluation — Keyword Hit Rate & Faithfulness")
    print(f"  Questions: {len(test_set)}   model={model}")
    print(f"{'='*70}\n")

    for i, entry in enumerate(test_set, start=1):
        question: str = entry["question"]
        keywords: list[str] = entry.get("answer_keywords", [])

        try:
            chunks = await retriever.retrieve(question, final_k=8)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:02d}] RETRIEVAL ERROR: {exc}")
            keyword_rates.append(0.0)
            faithfulness_verdicts.append(False)
            continue

        if not has_sufficient_context(chunks):
            answer = (
                "I don't have enough context in the indexed codebase to answer this confidently."
            )
        else:
            messages = build_prompt(question, chunks)
            try:
                tokens: list[str] = []
                async for token in stream_response_fn(messages, model):
                    tokens.append(token)
                answer = "".join(tokens)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i:02d}] GENERATION ERROR: {exc}")
                keyword_rates.append(0.0)
                faithfulness_verdicts.append(False)
                continue

        khr = _keyword_hit_rate_for_entry(answer, keywords)
        keyword_rates.append(khr)

        if run_judge and chunks:
            is_faithful = await _judge_faithfulness(
                question, answer, chunks, model, ollama_url
            )
        else:
            is_faithful = None  # skip judge — excluded from denominator
        faithfulness_verdicts.append(is_faithful)

        if is_faithful is None:
            faithful_label = "judge-skipped"
        else:
            faithful_label = "faithful" if is_faithful else "unfaithful"
        print(
            f"  [{i:02d}] khr={khr:.2f}  {faithful_label:<10} | "
            f"q={question[:55]!r}"
        )
        if khr < 1.0:
            missing = [kw for kw in keywords if kw.lower() not in answer.lower()]
            print(f"          missing keywords: {missing}")

    keyword_hit_rate = sum(keyword_rates) / len(keyword_rates) if keyword_rates else 0.0

    # Exclude None (judge skipped / network error) from the faithfulness denominator.
    judged = [v for v in faithfulness_verdicts if v is not None]
    faithfulness = sum(judged) / len(judged) if judged else 0.0
    judge_errors = len(faithfulness_verdicts) - len(judged)
    if judge_errors:
        print(f"  [warn] {judge_errors} judge call(s) skipped (network error or judge disabled)")

    print(f"\n{'='*70}")
    print(f"  keyword_hit_rate = {keyword_hit_rate:.3f}  (target > {_KEYWORD_HIT_RATE_TARGET})")
    print(f"  faithfulness     = {faithfulness:.3f}")
    print(
        f"  {'PASS' if keyword_hit_rate > _KEYWORD_HIT_RATE_TARGET else 'NEEDS TUNING'}"
    )
    print(f"{'='*70}\n")

    return {
        "keyword_hit_rate": keyword_hit_rate,
        "faithfulness": faithfulness,
        "n": len(test_set),
    }


# ---------------------------------------------------------------------------
# CLI entry point — requires a running stack
# ---------------------------------------------------------------------------


async def _main(qa_path: Path, repo_id: str, model: str, skip_judge: bool) -> None:
    from core.config import settings
    from core.dependencies import get_bm25_data, get_chroma_client, get_embed_service
    from retrieval.hybrid_retriever import HybridRetriever
    import retrieval.ollama_client as ollama_client_module

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

    metrics = await evaluate_generation(
        test_set=test_set,
        retriever=retriever,
        stream_response_fn=ollama_client_module.stream_response,
        model=model,
        ollama_url=str(settings.OLLAMA_URL).rstrip("/"),
        run_judge=not skip_judge,
    )
    sys.exit(0 if metrics["keyword_hit_rate"] > _KEYWORD_HIT_RATE_TARGET else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generation quality.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path(__file__).parent / "golden_qa" / "markupsafe_qa.json",
        help="Path to golden QA JSON file.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="8-char repo_id. If omitted, derived from markupsafe URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model for generation (default: qwen2.5-coder:7b).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-Judge faithfulness check (faster).",
    )
    args = parser.parse_args()

    if args.repo_id is None:
        import hashlib
        args.repo_id = hashlib.md5(
            b"https://github.com/pallets/markupsafe", usedforsecurity=False
        ).hexdigest()[:8]
        print(f"[info] Using derived repo_id={args.repo_id!r} for markupsafe")

    asyncio.run(_main(args.qa, args.repo_id, args.model, args.skip_judge))
