"""Helpers to detect text-readable files and read them with a byte cap."""
from __future__ import annotations

from pathlib import Path

_PROBE_BYTES = 8192
_MIN_TEXT_RATIO = 0.85


def is_probably_text(path: Path) -> bool:
    """Return True when *path* looks like UTF-8 text (not binary)."""
    try:
        sample = path.read_bytes()[:_PROBE_BYTES]
    except OSError:
        return False

    if not sample:
        return True

    if b"\x00" in sample:
        return False

    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass

    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in sample)
    return printable / len(sample) >= _MIN_TEXT_RATIO


def read_text_capped(path: Path, max_bytes: int) -> tuple[str, bool]:
    """Read up to *max_bytes* from *path* as UTF-8 text.

    Returns ``(text, truncated)`` where *truncated* is True when the file
    exceeded the byte limit.
    """
    size = path.stat().st_size
    truncated = size > max_bytes
    raw = path.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace"), truncated


def max_index_bytes() -> int:
    from core.config import settings

    return settings.MAX_INDEX_FILE_MB * 1024 * 1024
