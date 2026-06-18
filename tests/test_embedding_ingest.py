"""Unit tests for local embedding service and pipelined ingest orchestration."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.mark.asyncio
async def test_embed_batch_uses_ollama_when_configured() -> None:
    from ingestion.embedding_service import EmbeddingService

    fake_vectors = [[0.1] * 768, [0.2] * 768]

    with patch(
        "ingestion.embedding_service.settings.EMBED_BACKEND",
        "ollama",
    ), patch.object(
        EmbeddingService,
        "_embed_batch_ollama",
        new_callable=AsyncMock,
        return_value=fake_vectors,
    ) as mock_ollama:
        service = EmbeddingService()
        result = await service.embed_batch(["def foo(): pass", "class Bar: pass"])

    mock_ollama.assert_awaited_once()
    assert len(result) == 2
    assert all(len(vector) == 768 for vector in result)


@pytest.mark.asyncio
async def test_embed_batch_local_returns_768_dim_vectors() -> None:
    from ingestion.embedding_service import EmbeddingService

    fake_vectors = [[0.1] * 768, [0.2] * 768]

    with patch(
        "ingestion.embedding_service.settings.EMBED_BACKEND",
        "sentence_transformers",
    ), patch(
        "ingestion.embedding_service._get_local_model",
        return_value=MagicMock(),
    ), patch(
        "ingestion.embedding_service.EmbeddingService._encode_sentence_transformers",
        return_value=fake_vectors,
    ):
        service = EmbeddingService()
        result = await service.embed_batch(["def foo(): pass", "class Bar: pass"])

    assert len(result) == 2
    assert all(len(vector) == 768 for vector in result)


@pytest.mark.asyncio
async def test_orchestrator_pipelines_incremental_upserts(tmp_path: Path, monkeypatch) -> None:
    from core.config import settings

    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (tmp_path / "a.py").write_text("def placeholder(): pass")

    chunks = [
        {
            "text": f"def fn_{i}(): return {i}",
            "file_path": f"src/module_{i // 3}.py",
            "language": "python",
            "chunk_type": "function_definition",
            "start_line": i + 1,
            "end_line": i + 2,
            "symbol_name": f"fn_{i}",
        }
        for i in range(10)
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
        "ingestion.orchestrator.ASTChunker"
    ) as mock_chunker_cls, patch.object(
        embed_service,
        "embed_batch",
        new_callable=AsyncMock,
    ) as mock_embed, patch.object(
        chroma_writer,
        "upsert",
    ) as mock_upsert, patch.object(
        orchestrator._bm25,
        "build_index",
    ), patch(
        "core.dependencies.get_bm25_data",
    ):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = chunks
        mock_chunker_cls.return_value = mock_chunker

        async def _fake_embed(texts, **_: object) -> list[list[float]]:
            return [[float(i)] * 768 for i in range(len(texts))]

        mock_embed.side_effect = _fake_embed

        result = await orchestrator.ingest_repo(
            "https://github.com/example/repo",
            branch="main",
        )

    assert result["chunks_indexed"] == 11
    assert mock_embed.call_count >= 1
    assert mock_upsert.call_count >= 1
    for call in mock_embed.call_args_list:
        texts = call.args[0] if call.args else call.kwargs.get("texts", [])
        assert len(texts) <= settings.INGEST_FLUSH_SIZE


@pytest.mark.asyncio
async def test_orchestrator_flushes_in_multiple_batches(
    tmp_path: Path, monkeypatch
) -> None:
    from core.config import settings

    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("core.config.settings.INGEST_FLUSH_SIZE", 8)

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (tmp_path / "a.py").write_text("def placeholder(): pass")

    chunks = [
        {
            "text": f"def fn_{i}(): return {i}",
            "file_path": f"src/module_{i // 3}.py",
            "language": "python",
            "chunk_type": "function_definition",
            "start_line": i + 1,
            "end_line": i + 2,
            "symbol_name": f"fn_{i}",
        }
        for i in range(20)
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
    ) as mock_upsert, patch.object(
        orchestrator._bm25,
        "build_index",
    ), patch(
        "core.dependencies.get_bm25_data",
    ):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = chunks
        mock_chunker_cls.return_value = mock_chunker

        async def _fake_embed(texts, **_: object) -> list[list[float]]:
            return [[float(i)] * 768 for i in range(len(texts))]

        mock_embed.side_effect = _fake_embed

        result = await orchestrator.ingest_repo(
            "https://github.com/example/repo",
            branch="main",
        )

    assert result["chunks_indexed"] == 20
    assert mock_embed.call_count >= 2
    assert mock_upsert.call_count >= 2
    for call in mock_embed.call_args_list:
        texts = call.args[0] if call.args else call.kwargs.get("texts", [])
        assert len(texts) <= 8


@pytest.mark.asyncio
async def test_orchestrator_reports_embedding_phase_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (tmp_path / "a.py").write_text("def placeholder(): pass")

    chunks = [
        {
            "text": f"chunk {i}",
            "file_path": "a.py",
            "language": "python",
            "chunk_type": "function_definition",
            "start_line": 1,
            "end_line": 1,
            "symbol_name": f"fn_{i}",
        }
        for i in range(4)
    ]

    chroma_writer = ChromaWriter(chroma_path=str(chroma_dir))
    embed_service = EmbeddingService()
    orchestrator = IngestionOrchestrator(chroma_writer, embed_service)
    progress_events: list[tuple[str, float, str]] = []

    async def _capture(phase: str, progress: float, label: str) -> None:
        progress_events.append((phase, progress, label))

    with patch.object(
        orchestrator._cloner,
        "clone",
        return_value=(tmp_path, "abc123"),
    ), patch.object(
        orchestrator._walker,
        "walk_with_stats",
        return_value=([("a.py", str(tmp_path / "a.py"), "python")], []),
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

        async def _fake_embed_progress(texts, **_: object) -> list[list[float]]:
            return [[0.0] * 768 for _ in texts]

        mock_embed.side_effect = _fake_embed_progress

        await orchestrator.ingest_repo(
            "https://github.com/example/repo",
            progress_callback=_capture,
        )

    phases = [event[0] for event in progress_events]
    assert "chunking" in phases
    assert "embedding" in phases
    assert "bm25" in phases
    assert progress_events[-1][0] == "bm25"
    assert progress_events[-1][1] == 1.0


def test_job_store_tracks_phase() -> None:
    import jobs.job_store as job_store

    job_store.create_job("job-1", "repo123")
    job_store.update_job("job-1", progress=0.5, current_file="Embedding chunks 10/20", phase="embedding")

    job = job_store.get_job("job-1")
    assert job is not None
    assert job["phase"] == "embedding"
    assert job["current_file"] == "Embedding chunks 10/20"
    assert job["progress"] == 0.5
