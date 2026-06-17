"""Phase 2 unit tests — Hybrid Retrieval Pipeline.

All tests are fully mocked (no Ollama, no ChromaDB server required).
Run with:
    pytest tests/test_phase2_retrieval.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "def foo(): pass", idx: int = 0) -> dict:
    return {
        "text": text,
        "file_path": f"src/file{idx}.py",
        "language": "python",
        "chunk_type": "function_definition",
        "start_line": idx * 10 + 1,
        "end_line": idx * 10 + 5,
        "symbol_name": f"symbol_{idx}",
        "score": 0.5,
    }


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    def test_score_formula(self):
        """score must equal 1 - distance."""
        from retrieval.dense_retriever import DenseRetriever

        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "documents": [["def foo(): pass", "class Bar: pass"]],
            "metadatas": [
                [
                    {"file_path": "a.py", "language": "python", "chunk_type": "fn",
                     "start_line": 1, "end_line": 3, "symbol_name": "foo"},
                    {"file_path": "b.py", "language": "python", "chunk_type": "cls",
                     "start_line": 5, "end_line": 10, "symbol_name": "Bar"},
                ]
            ],
            "distances": [[0.1, 0.4]],
        }

        retriever = DenseRetriever()
        results = retriever.retrieve([0.0] * 768, mock_collection, k=10)

        assert len(results) == 2
        assert abs(results[0]["score"] - 0.9) < 1e-6, "score = 1 - 0.1"
        assert abs(results[1]["score"] - 0.6) < 1e-6, "score = 1 - 0.4"

    def test_metadata_passthrough(self):
        """All metadata keys must appear on each result dict."""
        from retrieval.dense_retriever import DenseRetriever

        meta = {
            "file_path": "src/main.py",
            "language": "python",
            "chunk_type": "function_definition",
            "start_line": 1,
            "end_line": 5,
            "symbol_name": "main",
        }
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["def main(): pass"]],
            "metadatas": [[meta]],
            "distances": [[0.2]],
        }

        results = DenseRetriever().retrieve([0.0] * 768, mock_collection)

        assert results[0]["file_path"] == "src/main.py"
        assert results[0]["symbol_name"] == "main"
        assert results[0]["text"] == "def main(): pass"

    def test_empty_collection_returns_empty_without_query(self):
        """When count=0 the query must NOT be called (n_results=0 crashes ChromaDB)."""
        from retrieval.dense_retriever import DenseRetriever

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        results = DenseRetriever().retrieve([0.0] * 768, mock_collection)

        assert results == []
        mock_collection.query.assert_not_called()


# ---------------------------------------------------------------------------
# SparseRetriever
# ---------------------------------------------------------------------------

class TestSparseRetriever:
    def _build_index_and_corpus(self) -> tuple[BM25Okapi, list[dict]]:
        # BM25 tokenizes via lowercase().split() so avoid parens/punctuation
        # that would make "authenticate" part of a larger token like "authenticate(user,"
        texts = [
            "authenticate user password login credentials",
            "class UserRepository save delete",
            "connect database url host port",
            "import os sys path",
        ]
        corpus = [_make_chunk(t, i) for i, t in enumerate(texts)]
        tokenized = [t.lower().split() for t in texts]
        index = BM25Okapi(tokenized)
        return index, corpus

    def test_top_k_limits_results(self):
        from retrieval.sparse_retriever import SparseRetriever

        index, corpus = self._build_index_and_corpus()
        results = SparseRetriever().retrieve("authenticate", index, corpus, k=2)

        assert len(results) <= 2

    def test_scores_are_descending(self):
        from retrieval.sparse_retriever import SparseRetriever

        index, corpus = self._build_index_and_corpus()
        results = SparseRetriever().retrieve("authenticate", index, corpus, k=10)

        scores = [s for s, _ in results]
        assert scores == sorted(scores, reverse=True), "Results must be score-descending"

    def test_relevant_chunk_is_top_result(self):
        from retrieval.sparse_retriever import SparseRetriever

        index, corpus = self._build_index_and_corpus()
        results = SparseRetriever().retrieve("authenticate user password", index, corpus, k=4)

        assert results, "Expected at least one result"
        top_chunk = results[0][1]
        assert "authenticate" in top_chunk["text"].lower()

    def test_zero_score_chunks_excluded(self):
        """Chunks with BM25 score <= 0 must not appear in results."""
        from retrieval.sparse_retriever import SparseRetriever

        corpus = [_make_chunk("hello world", 0)]
        tokenized = [["hello", "world"]]
        index = BM25Okapi(tokenized)

        # Query with completely unrelated term
        results = SparseRetriever().retrieve("zyxwvutsrqponm", index, corpus, k=10)
        assert results == []


# ---------------------------------------------------------------------------
# RRFFusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_deduplication_by_structural_key(self):
        """Chunks with same file_path:start_line:end_line must be merged into one entry."""
        from retrieval.rrf_fusion import merge

        # Same source location, slightly different text (e.g. from dense vs sparse cache)
        chunk_a = {**_make_chunk("def foo(): pass", idx=0), "file_path": "src/a.py", "start_line": 1, "end_line": 5}
        chunk_b = {**_make_chunk("def foo(): pass  # comment", idx=0), "file_path": "src/a.py", "start_line": 1, "end_line": 5}

        dense = [chunk_a]
        sparse = [(1.0, chunk_b)]

        results = merge(dense, sparse)
        assert len(results) == 1, "Same file_path:start_line:end_line must collapse to one entry"

    def test_deduplication_different_locations_not_merged(self):
        """Chunks at different source locations must NOT be merged."""
        from retrieval.rrf_fusion import merge

        chunk_a = {**_make_chunk("def foo(): pass", idx=0), "file_path": "src/a.py", "start_line": 1, "end_line": 5}
        chunk_b = {**_make_chunk("def bar(): pass", idx=1), "file_path": "src/a.py", "start_line": 10, "end_line": 15}

        dense = [chunk_a]
        sparse = [(1.0, chunk_b)]

        results = merge(dense, sparse)
        assert len(results) == 2, "Different source locations must remain separate"

    def test_rank_ordering_rrf_score(self):
        """Rank-1 in dense list must beat rank-50 regardless of BM25 score."""
        from retrieval.rrf_fusion import merge

        top_dense = _make_chunk("very relevant function", idx=0)
        far_chunk = _make_chunk("unrelated chunk xyz", idx=99)

        # top_dense is rank 1 in dense; far_chunk only appears in sparse at rank 1
        dense = [top_dense] + [_make_chunk(f"filler {i}", idx=i + 1) for i in range(49)]
        sparse = [(10.0, far_chunk)]

        results = merge(dense, sparse)
        top_key = results[0]["text"][:100]
        assert top_key == top_dense["text"][:100], (
            "Rank-1 dense chunk should beat single sparse chunk at low rank"
        )

    def test_both_lists_contribute(self):
        """Chunks appearing in both lists should have higher RRF score than
        chunks appearing in only one."""
        from retrieval.rrf_fusion import merge

        shared_text = "shared chunk content here"
        chunk_in_both = _make_chunk(shared_text, idx=0)
        chunk_dense_only = _make_chunk("dense only content xyz", idx=1)

        dense = [chunk_in_both, chunk_dense_only]
        sparse = [(5.0, chunk_in_both)]

        results = merge(dense, sparse)
        scores = {r["text"][:100]: r["rrf_score"] for r in results}

        assert scores[shared_text[:100]] > scores["dense only content xyz"[:100]]

    def test_returns_at_most_100(self):
        from retrieval.rrf_fusion import merge

        dense = [_make_chunk(f"chunk {i}", idx=i) for i in range(80)]
        sparse = [(float(i), _make_chunk(f"sparse {i}", idx=i + 100)) for i in range(80)]

        results = merge(dense, sparse)
        assert len(results) <= 100

    def test_empty_inputs(self):
        from retrieval.rrf_fusion import merge

        assert merge([], []) == []
        assert merge([_make_chunk()], []) != []
        assert merge([], [(1.0, _make_chunk())]) != []


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def test_top_k_slice(self):
        """rerank() must return at most top_k results."""
        from retrieval.reranker import Reranker

        candidates = [_make_chunk(f"candidate {i}", idx=i) for i in range(20)]
        fake_scores = list(range(20, 0, -1))  # descending

        reranker = Reranker()
        with patch("retrieval.reranker._get_cross_encoder") as mock_get_ce:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = np.array(fake_scores, dtype=float)
            mock_get_ce.return_value = mock_ce

            results = reranker.rerank("test query", candidates, top_k=5)

        assert len(results) == 5

    def test_score_attached_to_chunks(self):
        """Each returned chunk must have a ``score`` key with a sigmoid probability."""
        from retrieval.reranker import Reranker

        candidates = [_make_chunk("def foo(): pass", idx=0)]
        fake_prob = 0.73  # simulates a post-softmax probability

        with patch("retrieval.reranker._get_cross_encoder") as mock_get_ce:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = np.array([fake_prob], dtype=float)
            mock_get_ce.return_value = mock_ce

            results = Reranker().rerank("foo function", candidates, top_k=8)

        assert "score" in results[0]
        assert results[0]["score"] == 1.0  # single candidate normalised to certainty

    def test_logits_normalised_to_probabilities(self):
        """Raw cross-encoder logits must be softmax-normalised into [0, 1]."""
        from retrieval.reranker import Reranker

        candidates = [_make_chunk(f"fn {i}", idx=i) for i in range(3)]
        logits = np.array([-10.0, -11.0, -12.0], dtype=float)

        with patch("retrieval.reranker._get_cross_encoder") as mock_get_ce:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = logits
            mock_get_ce.return_value = mock_ce

            results = Reranker().rerank("query", candidates, top_k=3)

        assert mock_ce.predict.call_args.kwargs.get("apply_softmax") is False
        assert all(0.0 <= r["score"] <= 1.0 for r in results)
        assert abs(sum(r["score"] for r in results) - 1.0) < 1e-4
        assert results[0]["score"] > results[-1]["score"]

    def test_results_sorted_descending(self):
        from retrieval.reranker import Reranker

        candidates = [_make_chunk(f"fn {i}", idx=i) for i in range(5)]
        scores = [1.0, 4.0, 2.0, 5.0, 3.0]

        with patch("retrieval.reranker._get_cross_encoder") as mock_get_ce:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = np.array(scores, dtype=float)
            mock_get_ce.return_value = mock_ce

            results = Reranker().rerank("query", candidates, top_k=5)

        result_scores = [r["score"] for r in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_empty_candidates_returns_empty(self):
        from retrieval.reranker import Reranker

        results = Reranker().rerank("anything", [], top_k=8)
        assert results == []


# ---------------------------------------------------------------------------
# HybridRetriever (end-to-end mock)
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_full_pipeline_returns_top_k(self):
        """All pipeline stages wired together with mocked sub-components."""
        from retrieval.hybrid_retriever import HybridRetriever

        # Build a small BM25 corpus
        texts = [f"def function_{i}(): pass" for i in range(10)]
        corpus = [_make_chunk(t, i) for i, t in enumerate(texts)]
        tokenized = [t.lower().split() for t in texts]
        bm25_index = BM25Okapi(tokenized)

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 25
        mock_collection.query.return_value = {
            "documents": [[c["text"] for c in corpus[:5]]],
            "metadatas": [[
                {k: v for k, v in c.items() if k not in ("text", "score")}
                for c in corpus[:5]
            ]],
            "distances": [[0.1 * i for i in range(5)]],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_collection.return_value = mock_collection

        # Mock EmbeddingService
        mock_embed_service = MagicMock()
        mock_embed_service.embed_batch = AsyncMock(
            return_value=[[0.0] * 768]
        )

        retriever = HybridRetriever(
            collection_name="repo_test",
            chroma_client=mock_chroma_client,
            embed_service=mock_embed_service,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        # Mock the reranker to avoid loading the CrossEncoder model
        fake_reranked = [
            {**corpus[i], "score": float(5 - i)} for i in range(min(3, len(corpus)))
        ]
        with patch.object(retriever._reranker, "rerank", return_value=fake_reranked):
            results = await retriever.retrieve("function_0", final_k=3)

        assert len(results) <= 3
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_calls_embed_once(self):
        """EmbeddingService.embed_batch must be called exactly once per query."""
        from retrieval.hybrid_retriever import HybridRetriever

        texts = ["def foo(): pass"]
        corpus = [_make_chunk(texts[0], 0)]
        bm25_index = BM25Okapi([texts[0].lower().split()])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 25
        mock_collection.query.return_value = {
            "documents": [[texts[0]]],
            "metadatas": [[{k: v for k, v in corpus[0].items() if k not in ("text", "score")}]],
            "distances": [[0.1]],
        }
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.0] * 768])

        retriever = HybridRetriever(
            collection_name="repo_x",
            chroma_client=mock_chroma,
            embed_service=mock_embed,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        with patch.object(retriever._reranker, "rerank", return_value=corpus[:1]):
            await retriever.retrieve("foo")

        mock_embed.embed_batch.assert_called_once_with(["foo"], keep_alive="0")


# ---------------------------------------------------------------------------
# HybridRetriever — small-corpus fast path
# ---------------------------------------------------------------------------

class TestHybridRetrieverFastPath:
    @pytest.mark.asyncio
    async def test_small_corpus_returns_all_chunks(self):
        from retrieval.hybrid_retriever import HybridRetriever

        corpus = [_make_chunk(f"chunk {i}", idx=i) for i in range(13)]
        bm25_index = BM25Okapi([c["text"].lower().split() for c in corpus])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 13

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.0] * 768])

        retriever = HybridRetriever(
            collection_name="repo_small",
            chroma_client=mock_chroma,
            embed_service=mock_embed,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        with patch.object(retriever._reranker, "rerank") as mock_rerank:
            results = await retriever.retrieve("what's in the repo")

        assert len(results) == 13
        assert all(r["score"] == 1.0 for r in results)
        mock_embed.embed_batch.assert_not_called()
        mock_rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_small_corpus_skips_reranker(self):
        from retrieval.hybrid_retriever import HybridRetriever

        corpus = [_make_chunk("def foo(): pass", idx=0)]
        bm25_index = BM25Okapi([["def", "foo():", "pass"]])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.0] * 768])

        retriever = HybridRetriever(
            collection_name="repo_x",
            chroma_client=mock_chroma,
            embed_service=mock_embed,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        with patch.object(retriever._reranker, "rerank") as mock_rerank:
            await retriever.retrieve("explain foo")

        mock_rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_corpus_uses_full_pipeline(self):
        from retrieval.hybrid_retriever import HybridRetriever

        texts = [f"def function_{i}(): pass" for i in range(25)]
        corpus = [_make_chunk(t, i) for i, t in enumerate(texts)]
        bm25_index = BM25Okapi([t.lower().split() for t in texts])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 25
        mock_collection.query.return_value = {
            "documents": [[c["text"] for c in corpus[:5]]],
            "metadatas": [[
                {k: v for k, v in c.items() if k not in ("text", "score")}
                for c in corpus[:5]
            ]],
            "distances": [[0.1 * i for i in range(5)]],
        }

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.0] * 768])

        retriever = HybridRetriever(
            collection_name="repo_large",
            chroma_client=mock_chroma,
            embed_service=mock_embed,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        fake_reranked = [{**corpus[0], "score": 0.9}]
        with patch.object(retriever._reranker, "rerank", return_value=fake_reranked) as mock_rerank:
            await retriever.retrieve("function_0", final_k=1)

        mock_embed.embed_batch.assert_called_once()
        mock_rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_overview_query_includes_readme_first(self):
        from retrieval.hybrid_retriever import HybridRetriever

        corpus = [
            _make_chunk("def foo(): pass", idx=0),
            {**_make_chunk("# My Project", idx=1), "file_path": "README.md"},
        ]
        bm25_index = BM25Okapi([c["text"].lower().split() for c in corpus])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 25

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.0] * 768])

        retriever = HybridRetriever(
            collection_name="repo_overview",
            chroma_client=mock_chroma,
            embed_service=mock_embed,
            bm25_index=bm25_index,
            corpus=corpus,
        )

        results = await retriever.retrieve("what files are in this repo")

        assert results[0]["file_path"] == "README.md"
        mock_embed.embed_batch.assert_not_called()
