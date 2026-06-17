"""Embedding service backed by Ollama's nomic-embed-text model.

Design notes:
- Texts are sent to ``/api/embed`` in batches of ``EMBED_BATCH_SIZE``.
- One HTTP request per batch (no asyncio.gather fan-out) to match
  ``OLLAMA_NUM_PARALLEL=1`` in docker-compose.
- Individual texts are capped at 8 000 characters before being sent;
  nomic-embed-text has an 8 192-token context window and longer inputs
  are silently truncated by the model anyway.
"""
from __future__ import annotations

import asyncio

import httpx

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
_MAX_RETRIES = 3


class EmbeddingService:
    """Async wrapper around Ollama's /api/embed endpoint."""

    async def embed_batch(
        self,
        texts: list[str],
        *,
        keep_alive: str | None = None,
    ) -> list[list[float]]:
        """Embed *all* texts, batching into groups of ``EMBED_BATCH_SIZE``."""
        if not texts:
            return []

        keep_alive = keep_alive if keep_alive is not None else settings.EMBED_KEEP_ALIVE
        batch_size = settings.EMBED_BATCH_SIZE
        embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                prompts = [text[:8000] if text.strip() else " " for text in batch]
                logger.debug(
                    "Embedding batch %d–%d / %d",
                    batch_start + 1,
                    batch_start + len(batch),
                    len(texts),
                )
                vectors = await self._embed_batch_request(
                    client, prompts, keep_alive=keep_alive
                )
                embeddings.extend(vectors)

        return embeddings

    async def _embed_batch_request(
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
                if len(vectors) != len(prompts):
                    raise ValueError(
                        f"Ollama returned {len(vectors)} embeddings for "
                        f"{len(prompts)} inputs"
                    )
                if any(not vector for vector in vectors):
                    raise ValueError("Ollama returned an empty embedding vector")
                if settings.EMBED_DIM and any(
                    len(vector) != settings.EMBED_DIM for vector in vectors
                ):
                    raise ValueError(
                        f"Expected {settings.EMBED_DIM}-dim embeddings, "
                        f"got lengths {[len(v) for v in vectors]}"
                    )
                return vectors
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
                last_exc = exc
                if attempt + 1 < _MAX_RETRIES:
                    delay = 2**attempt
                    logger.warning(
                        "Embed batch failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
