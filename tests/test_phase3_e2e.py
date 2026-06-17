"""Phase 3 end-to-end tests — LLM Generation Layer.

Tests cover:
  - ``has_sufficient_context`` guardrail (short-circuits before LLM call)
  - ``build_prompt`` structure: system role, chunk headers, tiktoken trim
  - ``count_tokens`` accuracy
  - ``stream_response`` NDJSON parsing and done-detection (mocked HTTP)

No Ollama server or GPU is required — all external calls are mocked.

Run with:
    pytest tests/test_phase3_e2e.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str = "def foo(): pass",
    score: float = 0.8,
    idx: int = 0,
) -> dict:
    return {
        "text": text,
        "file_path": f"src/module_{idx}.py",
        "language": "python",
        "chunk_type": "function_definition",
        "start_line": idx * 10 + 1,
        "end_line": idx * 10 + 5,
        "symbol_name": f"foo_{idx}",
        "score": score,
    }


# ---------------------------------------------------------------------------
# has_sufficient_context guardrail
# ---------------------------------------------------------------------------

class TestHasSufficientContext:
    def test_empty_list_returns_false(self):
        from retrieval.prompt_builder import has_sufficient_context

        assert has_sufficient_context([]) is False

    def test_all_low_scores_returns_false(self):
        from retrieval.prompt_builder import has_sufficient_context

        # Both below the 0.10 threshold (post-softmax probabilities)
        chunks = [_make_chunk(score=0.05), _make_chunk(score=0.02)]
        assert has_sufficient_context(chunks) is False

    def test_one_chunk_above_threshold_returns_true(self):
        from retrieval.prompt_builder import has_sufficient_context

        chunks = [_make_chunk(score=0.05), _make_chunk(score=0.30)]
        assert has_sufficient_context(chunks) is True

    def test_small_corpus_always_passes_guardrail(self):
        from retrieval.prompt_builder import has_sufficient_context

        chunks = [_make_chunk(score=0.05)]
        assert has_sufficient_context(chunks, corpus_size=13) is True

    def test_exactly_at_threshold(self):
        """Score exactly equal to MIN_SIMILARITY_THRESHOLD (0.10) must pass."""
        from retrieval.prompt_builder import (
            MIN_SIMILARITY_THRESHOLD,
            has_sufficient_context,
        )

        # Confirm the calibrated threshold is a probability (post-softmax),
        # not a raw logit.
        assert 0.0 < MIN_SIMILARITY_THRESHOLD < 1.0, (
            "Threshold must be a sigmoid probability in (0, 1)"
        )
        chunks = [_make_chunk(score=MIN_SIMILARITY_THRESHOLD)]
        assert has_sufficient_context(chunks) is True

    def test_missing_score_key_treated_as_zero(self):
        from retrieval.prompt_builder import has_sufficient_context

        chunk = {
            "text": "some code",
            "file_path": "a.py",
            "language": "python",
            "chunk_type": "fn",
            "start_line": 1,
            "end_line": 5,
            "symbol_name": "fn",
            # No "score" key
        }
        assert has_sufficient_context([chunk]) is False


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        from retrieval.prompt_builder import count_tokens

        assert count_tokens("") == 0

    def test_simple_python_code(self):
        from retrieval.prompt_builder import count_tokens

        tokens = count_tokens("def foo(x: int) -> int:\n    return x + 1\n")
        assert tokens > 0

    def test_longer_text_has_more_tokens(self):
        from retrieval.prompt_builder import count_tokens

        short = count_tokens("hello")
        long_ = count_tokens("hello " * 100)
        assert long_ > short


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_two_messages(self):
        """Must return exactly [system, user] messages."""
        from retrieval.prompt_builder import build_prompt

        messages = build_prompt("What does foo do?", [_make_chunk()])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_chunk_header_format(self):
        """User message must contain [Chunk 1] File: ... header."""
        from retrieval.prompt_builder import build_prompt

        chunk = _make_chunk(text="def bar(): pass", idx=3)
        messages = build_prompt("explain bar", [chunk])
        user_content = messages[1]["content"]

        assert "[Chunk 1] File:" in user_content
        assert "src/module_3.py" in user_content

    def test_multiple_chunks_numbered(self):
        from retrieval.prompt_builder import build_prompt

        chunks = [_make_chunk(idx=i) for i in range(3)]
        messages = build_prompt("multi", chunks)
        user_content = messages[1]["content"]

        assert "[Chunk 1]" in user_content
        assert "[Chunk 2]" in user_content
        assert "[Chunk 3]" in user_content

    def test_query_appears_in_user_message(self):
        from retrieval.prompt_builder import build_prompt

        messages = build_prompt("how does authentication work?", [_make_chunk()])
        assert "how does authentication work?" in messages[1]["content"]

    def test_system_message_mentions_codebase_oracle(self):
        from retrieval.prompt_builder import build_prompt

        messages = build_prompt("q", [_make_chunk()])
        assert "CodeBase Oracle" in messages[0]["content"]

    def test_tiktoken_trim_drops_excess_chunks(self):
        """When total chunk text exceeds the context budget, later chunks are dropped."""
        from retrieval.prompt_builder import build_prompt, context_token_limit, count_tokens

        budget = context_token_limit()
        half_budget = budget // 2
        # Approximate: ~4 chars per token → half_budget * 4 chars
        big_text = "x = 1\n" * (half_budget // 2)
        big_chunk_1 = _make_chunk(text=big_text, idx=0, score=0.9)
        big_chunk_2 = _make_chunk(text=big_text, idx=1, score=0.8)
        tiny_chunk = _make_chunk(text="def small(): pass", idx=2, score=0.7)

        # Two big chunks together exceed budget; tiny chunk is third
        messages = build_prompt("something", [big_chunk_1, big_chunk_2, tiny_chunk])
        user_content = messages[1]["content"]

        total_tokens = count_tokens(big_text) + count_tokens(big_text)
        if total_tokens > budget:
            # The second big chunk must have been dropped
            assert "[Chunk 3]" not in user_content

    def test_no_chunks_still_returns_messages(self):
        """Empty chunk list must still produce valid messages."""
        from retrieval.prompt_builder import build_prompt

        messages = build_prompt("what?", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# stream_response (mocked httpx)
# ---------------------------------------------------------------------------

class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_tokens_yielded_from_ndjson(self):
        """Each non-done NDJSON line must yield its content token."""
        from retrieval.ollama_client import stream_response

        lines = [
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "Hello"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": " world"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # aiter_lines() is called without await and returns an async iterator
        # directly, so use MagicMock (not AsyncMock) to avoid coroutine wrapping.
        mock_response.aiter_lines = MagicMock(return_value=_aiter(lines))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        mock_outer_cm = MagicMock()
        mock_outer_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_outer_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("retrieval.ollama_client.httpx.AsyncClient", return_value=mock_outer_cm):
            tokens = []
            async for token in stream_response([{"role": "user", "content": "hi"}], "qwen2.5-coder:7b"):
                tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_stops_on_done(self):
        """Generator must stop when ``done: true`` is received."""
        from retrieval.ollama_client import stream_response

        lines = [
            json.dumps({"message": {"content": "tok1"}, "done": False}),
            json.dumps({"message": {"content": "tok2"}, "done": True}),
            json.dumps({"message": {"content": "should_not_appear"}, "done": False}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=_aiter(lines))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        mock_outer_cm = MagicMock()
        mock_outer_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_outer_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("retrieval.ollama_client.httpx.AsyncClient", return_value=mock_outer_cm):
            tokens = []
            async for token in stream_response([{"role": "user", "content": "q"}], "model"):
                tokens.append(token)

        assert "should_not_appear" not in tokens

    @pytest.mark.asyncio
    async def test_empty_content_tokens_skipped(self):
        """Lines with empty content must not emit tokens."""
        from retrieval.ollama_client import stream_response

        lines = [
            json.dumps({"message": {"content": ""}, "done": False}),
            json.dumps({"message": {"content": "real"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=_aiter(lines))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)

        mock_outer_cm = MagicMock()
        mock_outer_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_outer_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("retrieval.ollama_client.httpx.AsyncClient", return_value=mock_outer_cm):
            tokens = []
            async for token in stream_response([], "model"):
                tokens.append(token)

        assert tokens == ["real"]


# ---------------------------------------------------------------------------
# Helper: async iterator from a plain list
# ---------------------------------------------------------------------------

async def _aiter(items):
    """Yield items from a list as an async iterator (avoids shadowing builtins.aiter)."""
    for item in items:
        yield item
