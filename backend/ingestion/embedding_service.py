"""Embedding service — Ollama (primary), FastEmbed ONNX, or sentence-transformers.

Design notes:
- Default backend is Ollama HTTP (``nomic-embed-text``) to keep model RAM out
  of the backend process.
- FastEmbed and sentence-transformers are optional in-process backends.
- Nomic models require ``search_document:`` / ``search_query:`` task prefixes.
- Individual texts are capped at 8 000 characters before being sent.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Literal

import httpx
import numpy as np

from core.config import settings
from core.debug_log import log_timing
from core.logger import get_logger

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

EmbedTask = Literal["document", "query"]

EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
_MAX_RETRIES = 3
_MAX_TEXT_CHARS = 8000

_fastembed_model: TextEmbedding | None = None
_fastembed_lock = threading.Lock()
_fastembed_failed = False

_local_model: SentenceTransformer | None = None
_local_model_lock = threading.Lock()
_local_model_failed = False
_use_ollama_fallback = False


def _effective_batch_size() -> int:
    if settings.EMBED_BACKEND == "fastembed":
        return settings.EMBED_FAST_BATCH_SIZE
    # Never encode more than one flush window per call (avoids OOM in mem_limit containers)
    return min(settings.EMBED_LOCAL_BATCH_SIZE, max(settings.INGEST_FLUSH_SIZE, 1))


def _uses_nomic_prefixes(model_name: str) -> bool:
    return "nomic-embed" in model_name.lower()


def _apply_task_prefix(texts: list[str], task: EmbedTask, model_name: str) -> list[str]:
    if not _uses_nomic_prefixes(model_name):
        return texts
    prefix = "search_query: " if task == "query" else "search_document: "
    return [prefix + text for text in texts]


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


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        normalized.append(arr.tolist())
    return normalized


def _get_fastembed_model() -> TextEmbedding:
    global _fastembed_model, _fastembed_failed, _use_ollama_fallback  # noqa: PLW0603

    if _fastembed_failed:
        raise RuntimeError("FastEmbed model previously failed to load")

    if _fastembed_model is None:
        with _fastembed_lock:
            if _fastembed_model is None:
                try:
                    from fastembed import TextEmbedding

                    kwargs: dict = {"model_name": settings.EMBED_FAST_MODEL}
                    if settings.EMBED_FAST_THREADS > 0:
                        kwargs["threads"] = settings.EMBED_FAST_THREADS
                    logger.info(
                        "Loading FastEmbed model: %s (threads=%s)",
                        settings.EMBED_FAST_MODEL,
                        settings.EMBED_FAST_THREADS or "auto",
                    )
                    _fastembed_model = TextEmbedding(**kwargs)
                except Exception as exc:
                    _fastembed_failed = True
                    logger.warning("FastEmbed model failed to load: %s", exc)
                    raise
    return _fastembed_model


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
                        "Loading sentence-transformers model: %s (device=%s)",
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
                        "sentence-transformers model failed to load, "
                        "falling back to Ollama: %s",
                        exc,
                    )
                    raise
    return _local_model


class EmbeddingService:
    """Async wrapper around Ollama, FastEmbed, or sentence-transformers."""

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbedTask = "document",
        keep_alive: str | None = None,
    ) -> list[list[float]]:
        """Embed *texts* using the configured backend."""
        if not texts:
            return []

        if settings.EMBED_BACKEND == "ollama":
            return await self._embed_batch_ollama(texts, task=task, keep_alive=keep_alive)

        if _use_ollama_fallback:
            return await self._embed_batch_ollama(texts, task=task, keep_alive=keep_alive)

        if settings.EMBED_BACKEND == "fastembed":
            try:
                return await self._embed_batch_fastembed(texts, task=task)
            except RuntimeError:
                logger.warning("FastEmbed unavailable, trying sentence-transformers")

        try:
            return await self._embed_batch_sentence_transformers(texts, task=task)
        except RuntimeError:
            return await self._embed_batch_ollama(texts, task=task, keep_alive=keep_alive)

    async def _embed_batch_fastembed(
        self,
        texts: list[str],
        *,
        task: EmbedTask,
    ) -> list[list[float]]:
        batch_size = _effective_batch_size()
        model_name = settings.EMBED_FAST_MODEL
        embeddings: list[list[float]] = []
        t0 = time.perf_counter()

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            prompts = _prepare_prompts(batch)
            prompts = _apply_task_prefix(prompts, task, model_name)
            logger.debug(
                "Embedding batch %d–%d / %d (fastembed)",
                batch_start + 1,
                batch_start + len(batch),
                len(texts),
            )
            vectors = await asyncio.to_thread(
                self._encode_fastembed, prompts, batch_size
            )
            embeddings.extend(vectors)

        if settings.DEBUG_TIMING and embeddings:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rate = len(embeddings) / max((time.perf_counter() - t0), 1e-6)
            log_timing(
                "embed_fastembed",
                elapsed_ms,
                {
                    "chunks": len(embeddings),
                    "chunks_per_sec": round(rate, 2),
                    "task": task,
                },
            )

        return embeddings

    def _encode_fastembed(self, prompts: list[str], batch_size: int) -> list[list[float]]:
        model = _get_fastembed_model()
        vectors = list(model.embed(prompts, batch_size=batch_size))
        normalized = _normalize_vectors([vec.tolist() for vec in vectors])
        return _validate_vectors(normalized, len(prompts))

    async def _embed_batch_sentence_transformers(
        self,
        texts: list[str],
        *,
        task: EmbedTask,
    ) -> list[list[float]]:
        batch_size = _effective_batch_size()
        model_name = settings.EMBED_LOCAL_MODEL
        embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            prompts = _prepare_prompts(batch)
            prompts = _apply_task_prefix(prompts, task, model_name)
            logger.debug(
                "Embedding batch %d–%d / %d (sentence-transformers)",
                batch_start + 1,
                batch_start + len(batch),
                len(texts),
            )
            vectors = await asyncio.to_thread(
                self._encode_sentence_transformers, prompts, batch_size
            )
            embeddings.extend(vectors)

        return embeddings

    def _encode_sentence_transformers(
        self, prompts: list[str], batch_size: int
    ) -> list[list[float]]:
        model = _get_local_model()
        encoded = model.encode(
            prompts,
            batch_size=batch_size,
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
        task: EmbedTask,
        keep_alive: str | None,
    ) -> list[list[float]]:
        keep_alive = keep_alive if keep_alive is not None else settings.EMBED_KEEP_ALIVE
        batch_size = settings.EMBED_BATCH_SIZE
        embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                prompts = _prepare_prompts(batch)
                prompts = _apply_task_prefix(prompts, task, EMBED_MODEL)
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
