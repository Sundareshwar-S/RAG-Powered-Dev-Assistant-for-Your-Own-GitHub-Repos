"""Unit tests for structure query detection and retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _make_chunk(file_path: str, text: str = "def foo(): pass") -> dict:
    return {
        "text": text,
        "file_path": file_path,
        "language": "python",
        "chunk_type": "function_definition",
        "start_line": 1,
        "end_line": 1,
        "symbol_name": "foo",
    }


class TestQueryIntent:
    def test_structure_queries_detected(self) -> None:
        from retrieval.query_intent import is_structure_query

        assert is_structure_query("what files are in the backend folder")
        assert is_structure_query("whare are inside the backend folder")
        assert is_structure_query("list files in backend/")
        assert is_structure_query("what's the file structure")

    def test_code_questions_not_structure(self) -> None:
        from retrieval.query_intent import is_structure_query

        assert not is_structure_query("how does authentication work")
        assert not is_structure_query("explain the foo function")

    def test_file_content_queries_not_structure(self) -> None:
        from retrieval.query_intent import is_structure_query

        assert not is_structure_query("what's inside the readme.md file")
        assert not is_structure_query("what is inside README.md")
        assert not is_structure_query("show me app.py")
        assert not is_structure_query("contents of pipeline.py")

    def test_extract_target_file_case_insensitive(self) -> None:
        from retrieval.query_intent import extract_target_file

        paths = ["README.md", "app.py", "backend/config.py"]
        assert extract_target_file("what's inside readme.md", paths) == "README.md"
        assert extract_target_file("explain backend/config.py", paths) == "backend/config.py"

    def test_extract_folder_prefix(self) -> None:
        from retrieval.query_intent import extract_folder_prefix

        assert extract_folder_prefix("what files are in the backend folder") == "backend"
        assert extract_folder_prefix("inside the backend directory") == "backend"
        assert extract_folder_prefix("what's in readme") is None
        assert extract_folder_prefix("what's the file structure") is None
        assert extract_folder_prefix("what there in the file structure") is None

    def test_file_structure_returns_repo_wide_tree(self) -> None:
        from retrieval.structure_retriever import retrieve_structure_context

        corpus = [
            _make_chunk("backend/config.py"),
            _make_chunk("notebooks/train.ipynb"),
        ]

        results = retrieve_structure_context("what there in the file structure", corpus)

        assert len(results) == 1
        assert "config.py" in results[0]["text"]
        assert "train.ipynb" in results[0]["text"]
        assert "file/" not in results[0]["text"] or "No indexed files under file/" not in results[0]["text"]


class TestStructureRetriever:
    def test_folder_listing_includes_all_backend_files(self) -> None:
        from retrieval.structure_retriever import retrieve_structure_context

        corpus = [
            _make_chunk("backend/__init__.py"),
            _make_chunk("backend/config.py"),
            _make_chunk("backend/main.py"),
            _make_chunk("backend/ollama_client.py"),
            _make_chunk("backend/prompt.py"),
            _make_chunk("backend/schemas.py"),
            _make_chunk("frontend/App.jsx"),
        ]

        results = retrieve_structure_context(
            "what files are in the backend folder",
            corpus,
        )

        assert len(results) == 1
        text = results[0]["text"]
        assert "config.py" in text
        assert "main.py" in text
        assert "ollama_client.py" in text
        assert "schemas.py" in text
        assert "App.jsx" not in text

    def test_repo_wide_tree_from_manifest(self) -> None:
        from retrieval.structure_retriever import retrieve_structure_context

        manifest = {
            "text": "Repository file index:\nbackend/\n  main.py",
            "file_path": "__manifest__/repo_tree.txt",
            "language": "plaintext",
            "chunk_type": "file_manifest",
            "start_line": 1,
            "end_line": 3,
            "symbol_name": "",
        }
        corpus = [_make_chunk("backend/main.py"), manifest]

        results = retrieve_structure_context("what's in this repo", corpus)

        assert results[0]["chunk_type"] == "file_manifest"


class TestHybridRetrieverStructurePath:
    @pytest.mark.asyncio
    async def test_structure_query_returns_compact_index(self) -> None:
        from retrieval.hybrid_retriever import HybridRetriever

        corpus = [
            _make_chunk("backend/config.py"),
            _make_chunk("backend/main.py"),
        ]
        bm25_index = BM25Okapi([c["text"].lower().split() for c in corpus])

        mock_collection = MagicMock()
        mock_collection.count.return_value = 268

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

        results = await retriever.retrieve("what files are in the backend folder")

        assert len(results) <= 3
        assert results[0]["chunk_type"] == "file_manifest"
        assert "config.py" in results[0]["text"]
        mock_embed.embed_batch.assert_not_called()
