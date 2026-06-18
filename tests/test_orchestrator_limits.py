"""Unit tests for orchestrator chunk safety limits."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _make_chunk(file_path: str, index: int) -> dict:
    return {
        "text": f"chunk {index} in {file_path}",
        "file_path": file_path,
        "language": "python",
        "chunk_type": "sliding_window",
        "start_line": index + 1,
        "end_line": index + 2,
        "symbol_name": f"chunk_{index}",
    }


@pytest.mark.asyncio
async def test_orchestrator_stops_at_total_chunk_cap(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("core.config.settings.INGEST_MAX_TOTAL_CHUNKS", 6)
    monkeypatch.setattr("core.config.settings.INGEST_FLUSH_SIZE", 100)

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()

    files = [
        (f"file_{i}.py", str(tmp_path / f"file_{i}.py"), "python")
        for i in range(5)
    ]
    for rel_path, full_path, _ in files:
        Path(full_path).write_text(f"# {rel_path}\n")

    chroma_writer = ChromaWriter(chroma_path=str(chroma_dir))
    embed_service = EmbeddingService()
    orchestrator = IngestionOrchestrator(chroma_writer, embed_service)

    def _chunk_side_effect(_source: str, rel_path: str, _language: str) -> list[dict]:
        return [_make_chunk(rel_path, 0), _make_chunk(rel_path, 1)]

    with patch.object(
        orchestrator._cloner,
        "clone",
        return_value=(tmp_path, "abc123"),
    ), patch.object(
        orchestrator._walker,
        "walk_with_stats",
        return_value=(files, []),
    ), patch(
        "ingestion.orchestrator._chunk_file",
        side_effect=_chunk_side_effect,
    ), patch.object(
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
        async def _fake_embed(texts, **_: object) -> list[list[float]]:
            return [[0.0] * 768 for _ in texts]

        mock_embed.side_effect = _fake_embed

        result = await orchestrator.ingest_repo(
            "https://github.com/example/repo",
            branch="main",
        )

    # Cap is 6 content chunks + 1 manifest chunk
    assert result["chunks_indexed"] == 7
    assert result["files_indexed"] == 3


@pytest.mark.asyncio
async def test_orchestrator_reports_current_file_before_chunking(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("core.config.settings.BM25_CACHE_DIR", str(tmp_path))

    from ingestion.chroma_writer import ChromaWriter
    from ingestion.embedding_service import EmbeddingService
    from ingestion.orchestrator import IngestionOrchestrator

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (tmp_path / "train.py").write_text("x = 1\n")

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
        return_value=([("train.py", str(tmp_path / "train.py"), "python")], []),
    ), patch(
        "ingestion.orchestrator._chunk_file",
        return_value=[_make_chunk("train.py", 0)],
    ), patch.object(
        embed_service,
        "embed_batch",
        new_callable=AsyncMock,
        return_value=[[0.0] * 768],
    ), patch.object(
        chroma_writer,
        "upsert",
    ), patch.object(
        orchestrator._bm25,
        "build_index",
    ), patch(
        "core.dependencies.get_bm25_data",
    ):
        await orchestrator.ingest_repo(
            "https://github.com/example/repo",
            progress_callback=_capture,
        )

    chunking_labels = [label for phase, _, label in progress_events if phase == "chunking"]
    assert any("Chunking train.py (1/1)" in label for label in chunking_labels)
