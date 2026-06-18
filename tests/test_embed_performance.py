"""Unit tests for embedding service, embed cache, and ingest performance."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.mark.asyncio
async def test_embed_batch_fastembed_returns_768_dim_vectors() -> None:
    from ingestion.embedding_service import EmbeddingService

    fake_vectors = [[0.1] * 768]

    with patch(
        "ingestion.embedding_service.settings.EMBED_BACKEND",
        "fastembed",
    ), patch(
        "ingestion.embedding_service.EmbeddingService._encode_fastembed",
        return_value=fake_vectors,
    ):
        service = EmbeddingService()
        result = await service.embed_batch(["def foo(): pass"], task="document")

    assert len(result) == 1
    assert len(result[0]) == 768


def test_nomic_prefix_applied_for_documents() -> None:
    from ingestion.embedding_service import _apply_task_prefix

    prefixed = _apply_task_prefix(
        ["hello"],
        "document",
        "nomic-ai/nomic-embed-text-v1.5-Q",
    )
    assert prefixed[0].startswith("search_document: ")


def test_nomic_prefix_applied_for_queries() -> None:
    from ingestion.embedding_service import _apply_task_prefix

    prefixed = _apply_task_prefix(
        ["hello"],
        "query",
        "nomic-ai/nomic-embed-text-v1.5-Q",
    )
    assert prefixed[0].startswith("search_query: ")


def test_embed_cache_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))

    from ingestion.embed_cache import (
        compute_content_hash,
        get_cached_vector,
        store_cached_vector,
    )

    text = "File: main.py\ndef foo(): pass"
    content_hash = compute_content_hash(text)
    vector = [0.1] * 768

    assert get_cached_vector(content_hash) is None
    store_cached_vector(content_hash, vector)
    assert get_cached_vector(content_hash) == vector

    cache_file = tmp_path / "embed_cache" / f"{content_hash}.json"
    assert cache_file.is_file()
    assert json.loads(cache_file.read_text())["vector"] == vector


@pytest.mark.asyncio
async def test_orchestrator_uses_embed_cache_on_reindex(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embed_cache import compute_content_hash, store_cached_vector
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (tmp_path / "a.py").write_text("def placeholder(): pass")

    embed_text = "File: a.py\ndef fn_0(): return 0"
    store_cached_vector(compute_content_hash(embed_text), [0.5] * 768)

    chunks = [
        {
            "text": "def fn_0(): return 0",
            "file_path": "a.py",
            "language": "python",
            "chunk_type": "function_definition",
            "start_line": 1,
            "end_line": 1,
            "symbol_name": "fn_0",
        }
    ]

    chroma_writer = ChromaWriter(chroma_path=str(chroma_dir))
    embed_service = EmbeddingService()
    orchestrator = IngestionOrchestrator(chroma_writer, embed_service)

    with patch.object(
        orchestrator._cloner,
        "clone",
        return_value=(tmp_path, "abc123"),
    ), patch.object(
        orchestrator._walker,
        "walk_with_stats",
        return_value=([("a.py", str(tmp_path / "a.py"), "python")], []),
    ), patch(
        "ingestion.orchestrator.build_manifest_chunks",
        return_value=[],
    ), patch(
        "ingestion.orchestrator.ASTChunker"
    ) as mock_chunker_cls, patch.object(
        embed_service,
        "embed_batch",
        new_callable=AsyncMock,
    ) as mock_embed, patch.object(
        chroma_writer,
        "upsert",
    ), patch.object(
        orchestrator._bm25,
        "build_index",
    ), patch(
        "core.dependencies.get_bm25_data",
    ):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = chunks
        mock_chunker_cls.return_value = mock_chunker

        await orchestrator.ingest_repo("https://github.com/example/repo")

    mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_embed_throughput_smoke() -> None:
    """Smoke benchmark: FastEmbed path should embed 20 chunks without error."""
    from ingestion.embedding_service import EmbeddingService

    texts = [f"def fn_{i}(): return {i}" for i in range(20)]
    fake_vectors = [[float(i)] * 768 for i in range(20)]

    with patch(
        "ingestion.embedding_service.settings.EMBED_BACKEND",
        "fastembed",
    ), patch(
        "ingestion.embedding_service.EmbeddingService._encode_fastembed",
        return_value=fake_vectors,
    ):
        service = EmbeddingService()
        t0 = time.perf_counter()
        result = await service.embed_batch(texts, task="document")
        elapsed = time.perf_counter() - t0

    assert len(result) == 20
    assert elapsed < 5.0
