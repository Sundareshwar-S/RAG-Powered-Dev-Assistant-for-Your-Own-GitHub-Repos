"""Embedding service backed by local sentence-transformers (primary) or Ollama.

Design notes:
- Ingest and query both use the same local ``SentenceTransformer`` path when
  available (``nomic-ai/nomic-embed-text-v1``, 768-dim, normalized).
- Ollama ``/api/embed`` is a fallback only if the local model fails to load.
- Sync ``encode()`` runs in a thread pool via ``asyncio.to_thread``.
- Individual texts are capped at 8 000 characters before being sent.
"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import httpx
import numpy as np

from core.config import settings
from core.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
_MAX_RETRIES = 3
_MAX_TEXT_CHARS = 8000

_local_model: SentenceTransformer | None = None
_local_model_lock = threading.Lock()
_local_model_failed = False
_use_ollama_fallback = False


def _get_local_model() -> SentenceTransformer:
    global _local_model, _local_model_failed, _use_ollama_fallback  # noqa: PLW0603

    if _local_model_failed:
        raise RuntimeError("Local embedding model previously failed to load")

    if _local_model is None:
        with _local_model_lock:
            if _local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info(
                        "Loading local embedding model: %s (device=%s)",
                        settings.EMBED_LOCAL_MODEL,
                        settings.EMBED_LOCAL_DEVICE,
                    )
                    _local_model = SentenceTransformer(
                        settings.EMBED_LOCAL_MODEL,
                        device=settings.EMBED_LOCAL_DEVICE,
                        trust_remote_code=True,
                    )
                except Exception as exc:
                    _local_model_failed = True
                    _use_ollama_fallback = True
                    logger.warning(
                        "Local embedding model failed to load, "
                        "falling back to Ollama: %s",
                        exc,
                    )
                    raise
    return _local_model


def _prepare_prompts(texts: list[str]) -> list[str]:
    return [text[:_MAX_TEXT_CHARS] if text.strip() else " " for text in texts]


def _validate_vectors(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, got {len(vectors)}"
        )
    if any(not vector for vector in vectors):
        raise ValueError("Received an empty embedding vector")
    if settings.EMBED_DIM and any(
        len(vector) != settings.EMBED_DIM for vector in vectors
    ):
        raise ValueError(
            f"Expected {settings.EMBED_DIM}-dim embeddings, "
            f"got lengths {[len(v) for v in vectors]}"
        )
    return vectors


class EmbeddingService:
    """Async wrapper around local sentence-transformers or Ollama /api/embed."""

    async def embed_batch(
        self,
        texts: list[str],
        *,
        keep_alive: str | None = None,
    ) -> list[list[float]]:
        """Embed *all* texts in batches of ``EMBED_LOCAL_BATCH_SIZE``."""
        if not texts:
            return []

        if _use_ollama_fallback:
            return await self._embed_batch_ollama(texts, keep_alive=keep_alive)

        try:
            return await self._embed_batch_local(texts)
        except RuntimeError:
            return await self._embed_batch_ollama(texts, keep_alive=keep_alive)

    async def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        batch_size = settings.EMBED_LOCAL_BATCH_SIZE
        embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            prompts = _prepare_prompts(batch)
            logger.debug(
                "Embedding batch %d–%d / %d (local)",
                batch_start + 1,
                batch_start + len(batch),
                len(texts),
            )
            vectors = await asyncio.to_thread(self._encode_local, prompts)
            embeddings.extend(vectors)

        return embeddings

    def _encode_local(self, prompts: list[str]) -> list[list[float]]:
        model = _get_local_model()
        encoded = model.encode(
            prompts,
            batch_size=settings.EMBED_LOCAL_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = np.asarray(encoded, dtype=float).tolist()
        return _validate_vectors(vectors, len(prompts))

    async def _embed_batch_ollama(
        self,
        texts: list[str],
        *,
        keep_alive: str | None,
    ) -> list[list[float]]:
        keep_alive = keep_alive if keep_alive is not None else settings.EMBED_KEEP_ALIVE
        batch_size = settings.EMBED_BATCH_SIZE
        embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                prompts = _prepare_prompts(batch)
                logger.debug(
                    "Embedding batch %d–%d / %d (ollama)",
                    batch_start + 1,
                    batch_start + len(batch),
                    len(texts),
                )
                vectors = await self._embed_batch_request_ollama(
                    client, prompts, keep_alive=keep_alive
                )
                embeddings.extend(vectors)

        return embeddings

    async def _embed_batch_request_ollama(
        self,
        client: httpx.AsyncClient,
        prompts: list[str],
        *,
        keep_alive: str,
    ) -> list[list[float]]:
        payload = {
            "model": EMBED_MODEL,
            "input": prompts,
            "truncate": True,
            "keep_alive": keep_alive,
        }
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.post(
                    f"{settings.OLLAMA_URL}/api/embed",
                    json=payload,
                )
                response.raise_for_status()
                vectors = response.json().get("embeddings", [])
                return _validate_vectors(vectors, len(prompts))
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
                last_exc = exc
                if attempt + 1 < _MAX_RETRIES:
                    delay = 2**attempt
                    logger.warning(
                        "Ollama embed batch failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
