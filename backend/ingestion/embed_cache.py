"""Content-hash cache for embedding vectors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

_HASH_CHUNK_SIZE = 65536


def compute_content_hash(text: str) -> str:
    """Return SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_dir() -> Path:
    return Path(settings.BM25_CACHE_DIR) / "embed_cache"


def get_cached_vector(content_hash: str) -> list[float] | None:
    """Return a cached embedding vector or ``None`` on miss."""
    path = _cache_dir() / f"{content_hash}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        vector = data.get("vector")
        if isinstance(vector, list) and vector:
            return [float(v) for v in vector]
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Corrupt embed cache entry %s: %s", content_hash[:12], exc)
    return None


def store_cached_vector(content_hash: str, vector: list[float]) -> None:
    """Persist *vector* under *content_hash*."""
    directory = _cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{content_hash}.json"
    path.write_text(
        json.dumps({"vector": vector}, ensure_ascii=False),
        encoding="utf-8",
    )
