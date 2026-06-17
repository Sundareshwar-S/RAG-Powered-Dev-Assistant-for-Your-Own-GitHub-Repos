"""Unit tests for local embedding service and pipelined ingest orchestration."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.mark.asyncio
async def test_embed_batch_local_returns_768_dim_vectors() -> None:
    from ingestion.embedding_service import EmbeddingService

    fake_vectors = [[0.1] * 768, [0.2] * 768]

    with patch(
        "ingestion.embedding_service._get_local_model",
        return_value=MagicMock(),
    ), patch(
        "ingestion.embedding_service.EmbeddingService._encode_local",
        return_value=fake_vectors,
    ):
        service = EmbeddingService()
        result = await service.embed_batch(["def foo(): pass", "class Bar: pass"])

    assert len(result) == 2
    assert all(len(vector) == 768 for vector in result)


@pytest.mark.asyncio
async def test_orchestrator_pipelines_incremental_upserts(tmp_path: Path) -> None:
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
        "walk",
        return_value=[("a.py", str(tmp_path / "a.py"), "python")],
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
    assert mock_upsert.call_count == mock_embed.call_count


@pytest.mark.asyncio
async def test_orchestrator_reports_embedding_phase_progress(tmp_path: Path) -> None:
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
        "walk",
        return_value=[("a.py", str(tmp_path / "a.py"), "python")],
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
