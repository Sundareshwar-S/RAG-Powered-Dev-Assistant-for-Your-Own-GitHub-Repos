"""Unit tests for documentation and HTML chunking limits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _long_lines(count: int, prefix: str = "line") -> str:
    return "\n".join(f"{prefix} {i}" for i in range(1, count + 1))


class TestDocChunking:
    def test_readme_produces_more_than_three_chunks(self) -> None:
        from ingestion.ast_chunker import ASTChunker

        source = _long_lines(200, "readme")
        chunks = ASTChunker("markdown").chunk(source, "README.md")

        assert len(chunks) > 3
        assert chunks[0]["start_line"] == 1
        assert chunks[-1]["end_line"] == 200

    def test_html_respects_sliding_cap(self) -> None:
        from ingestion.ast_chunker import ASTChunker

        source = _long_lines(200, "html")
        chunks = ASTChunker("plaintext").chunk(source, "templates/index.html")

        assert len(chunks) > 3
        assert len(chunks) <= 20

    def test_large_text_respects_max_chunks_per_file(self, monkeypatch) -> None:
        from ingestion.ast_chunker import ASTChunker

        monkeypatch.setattr("core.config.settings.MAX_CHUNKS_PER_FILE", 80)
        monkeypatch.setattr("core.config.settings.AST_MAX_SLIDING_CHUNKS", 0)
        monkeypatch.setattr("core.config.settings.HTML_MAX_SLIDING_CHUNKS", 0)

        source = _long_lines(5000, "csv")
        chunks = ASTChunker("plaintext").chunk(source, "dataset/train.csv")

        assert len(chunks) == 80

    def test_yaml_doc_is_fully_chunked(self) -> None:
        from ingestion.ast_chunker import ASTChunker

        source = _long_lines(150, "key")
        chunks = ASTChunker("plaintext").chunk(source, "config.yaml")

        assert len(chunks) > 3

    def test_python_script_produces_more_than_three_chunks(self) -> None:
        from ingestion.ast_chunker import ASTChunker

        source = _long_lines(200, "x =")
        chunks = ASTChunker("python").chunk(source, "train.py")

        assert len(chunks) > 3
        assert chunks[0]["chunk_type"] == "sliding_window"
