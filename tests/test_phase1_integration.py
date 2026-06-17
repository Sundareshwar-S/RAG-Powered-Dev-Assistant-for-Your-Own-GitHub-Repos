"""Phase 1 integration test — Step 1.11.

Runs the full ingestion pipeline against a small public GitHub repository
(pallets/markupsafe) and verifies that:
  - ChromaDB collection count > 0
  - BM25 cache file was written under BM25_CACHE_DIR
  - Returned metadata is consistent

Requirements to run
-------------------
- Internet access to clone from GitHub.
- Local embedding model ``nomic-ai/nomic-embed-text-v1`` (pre-downloaded in
  the backend Docker image, or pulled on first run when testing on host).

Run with:
    pytest tests/test_phase1_integration.py -v -s -m integration
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

TEST_REPO_URL = "https://github.com/pallets/markupsafe"
TEST_BRANCH = "main"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ingestion_pipeline(tmp_path: Path) -> None:
    """Clone markupsafe, ingest it, and assert ChromaDB + BM25 have data."""
    import chromadb

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    bm25_dir = tmp_path / "bm25_cache"
    bm25_dir.mkdir()
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()

    # Override paths via env so that config.py and bm25_builder pick them up
    os.environ["BM25_CACHE_DIR"] = str(bm25_dir)

    chroma_writer = ChromaWriter(chroma_path=str(chroma_dir))
    embed_service = EmbeddingService()
    orchestrator = IngestionOrchestrator(chroma_writer, embed_service)

    result = await orchestrator.ingest_repo(TEST_REPO_URL, TEST_BRANCH)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert result["repo_id"], "repo_id must be non-empty"
    assert result["chunks_indexed"] > 0, "At least one chunk must be indexed"
    assert result["collection"].startswith("repo_"), "Collection name format"

    collection_name = result["collection"]

    # ChromaDB has records
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)
    count = collection.count()
    assert count > 0, f"Expected chunks in ChromaDB, got {count}"
    assert count == result["chunks_indexed"], (
        f"ChromaDB count ({count}) must equal chunks_indexed "
        f"({result['chunks_indexed']})"
    )

    # BM25 cache file exists
    cache_file = bm25_dir / f"{collection_name}.json"
    assert cache_file.is_file(), f"BM25 cache file not found: {cache_file}"
    assert cache_file.stat().st_size > 0, "BM25 cache file must not be empty"

    # Sample chunk has expected metadata keys
    sample = collection.get(limit=1, include=["documents", "metadatas"])
    assert sample["documents"], "Expected at least one document in sample"
    meta = sample["metadatas"][0]
    for key in ("file_path", "language", "chunk_type", "start_line", "end_line"):
        assert key in meta, f"Metadata key '{key}' missing from chunk"

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    assert any(m.get("chunk_type") == "file_manifest" for m in all_meta), (
        "Expected at least one file_manifest chunk after ingest"
    )

    print(
        f"\n[PASS] Indexed {count} chunks from {TEST_REPO_URL} "
        f"into '{collection_name}'"
    )
    print(f"       Sample chunk: {meta['file_path']} "
          f"L{meta['start_line']}–{meta['end_line']} ({meta['chunk_type']})")


@pytest.mark.integration
def test_ast_chunker_python_produces_named_chunks() -> None:
    """Verify ASTChunker extracts named functions from a Python snippet."""
    from ingestion.ast_chunker import ASTChunker

    source = """
def hello(name: str) -> str:
    return f"Hello, {name}"


class Greeter:
    def greet(self) -> None:
        print("hi")
"""
    chunker = ASTChunker("python")
    chunks = chunker.chunk(source, "test.py")

    names = {c["symbol_name"] for c in chunks}
    assert "hello" in names, f"Expected 'hello' in chunk names, got {names}"
    assert "Greeter" in names, f"Expected 'Greeter' in chunk names, got {names}"

    for chunk in chunks:
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
        assert chunk["text"].strip()


@pytest.mark.integration
def test_file_walker_skips_excluded_dirs(tmp_path: Path) -> None:
    """FileWalker must skip .git, node_modules, __pycache__, venv, .venv."""
    from ingestion.file_walker import FileWalker

    # Create real .py files
    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "lib.py").write_text("y = 2")

    # Create files that should be skipped
    for skip in (".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"):
        skip_dir = tmp_path / skip
        skip_dir.mkdir()
        (skip_dir / "evil.py").write_text("evil = True")

    walker = FileWalker()
    results = walker.walk(tmp_path)
    found_paths = [r[0] for r in results]

    assert len(found_paths) == 2, f"Expected 2 files, got: {found_paths}"
    assert all(
        not any(skip in p for skip in (".git", "node_modules", "__pycache__", "venv", "dist"))
        for p in found_paths
    )


@pytest.mark.integration
def test_bm25_builder_cache_round_trip(tmp_path: Path) -> None:
    """BM25 cache write → reload produces the same tokenized corpus."""
    import json
    import os

    os.environ["BM25_CACHE_DIR"] = str(tmp_path)

    from ingestion.bm25_builder import BM25Builder

    # Fake a ChromaDB collection via a mock
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["def foo(): pass", "class Bar: pass"],
        "metadatas": [
            {"file_path": "a.py", "language": "python", "chunk_type": "function_definition",
             "start_line": 1, "end_line": 1, "symbol_name": "foo"},
            {"file_path": "a.py", "language": "python", "chunk_type": "class_definition",
             "start_line": 2, "end_line": 2, "symbol_name": "Bar"},
        ],
    }
    mock_client.get_collection.return_value = mock_collection

    builder = BM25Builder()

    # First call: cache miss → build
    index1, corpus1 = builder.build_index("repo_test01", mock_client)
    cache_file = tmp_path / "repo_test01.json"
    assert cache_file.is_file()

    # Second call: cache hit → load
    index2, corpus2 = builder.build_index("repo_test01", mock_client)

    assert len(corpus1) == len(corpus2) == 2
    assert corpus1[0]["text"] == corpus2[0]["text"]
    # index2 was loaded from cache; smoke-test a query
    scores = index2.get_scores(["foo"])
    assert len(scores) == 2
