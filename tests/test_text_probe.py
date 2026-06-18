"""Unit tests for text file detection and capped reads."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def test_is_probably_text_accepts_utf8(tmp_path: Path) -> None:
    from ingestion.text_probe import is_probably_text

    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello world\n", encoding="utf-8")

    assert is_probably_text(text_file) is True


def test_is_probably_text_rejects_binary(tmp_path: Path) -> None:
    from ingestion.text_probe import is_probably_text

    binary_file = tmp_path / "model.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff")

    assert is_probably_text(binary_file) is False


def test_read_text_capped_truncates_large_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.config.settings.MAX_INDEX_FILE_MB", 1)

    from ingestion.text_probe import max_index_bytes, read_text_capped

    big = tmp_path / "big.csv"
    big.write_text("a" * (max_index_bytes() + 1000))

    text, truncated = read_text_capped(big, max_index_bytes())

    assert truncated is True
    assert len(text.encode("utf-8")) <= max_index_bytes()
