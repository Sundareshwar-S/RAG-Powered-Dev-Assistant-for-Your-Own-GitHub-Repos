"""Unit tests for file-targeted retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _make_chunk(
    file_path: str,
    text: str,
    *,
    start_line: int = 1,
    end_line: int = 1,
) -> dict:
    return {
        "text": text,
        "file_path": file_path,
        "language": "markdown",
        "chunk_type": "sliding_window",
        "start_line": start_line,
        "end_line": end_line,
        "symbol_name": "",
    }


class TestFileTargetRetriever:
    def test_returns_all_readme_chunks(self) -> None:
        from retrieval.file_target_retriever import retrieve_file_target_context

        corpus = [
            _make_chunk("README.md", "# Title", start_line=1, end_line=5),
            _make_chunk("README.md", "Body paragraph", start_line=6, end_line=20),
            _make_chunk("app.py", "x = 1"),
        ]

        results = retrieve_file_target_context("what's inside readme.md", corpus)

        assert len(results) == 2
        assert all(r["file_path"] == "README.md" for r in results)
        assert results[0]["text"] == "# Title"
        assert results[1]["text"] == "Body paragraph"

    def test_case_insensitive_match(self) -> None:
        from retrieval.file_target_retriever import retrieve_file_target_context

        corpus = [_make_chunk("README.md", "Project docs")]

        results = retrieve_file_target_context("contents of readme.md", corpus)

        assert len(results) == 1
        assert results[0]["text"] == "Project docs"


class TestHybridRetrieverFileTargetPath:
    @pytest.mark.asyncio
    async def test_file_query_skips_embed_and_returns_all_chunks(self) -> None:
        from retrieval.hybrid_retriever import HybridRetriever

        corpus = [
            _make_chunk("README.md", "Section A", start_line=1, end_line=10),
            _make_chunk("README.md", "Section B", start_line=11, end_line=20),
            _make_chunk("app.py", "x = 1"),
        ]
        bm25_index = BM25Okapi([c["text"].lower().split() for c in corpus])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100

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

        results = await retriever.retrieve("what's inside the readme.md file")

        assert len(results) == 2
        assert all(r["file_path"] == "README.md" for r in results)
        mock_embed.embed_batch.assert_not_called()
