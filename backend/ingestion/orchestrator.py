"""Ingestion orchestrator — ties together all Phase 1 pipeline steps.

Pipeline
--------
1. Clone / pull the repo via :class:`GitCloner`.
2. Walk all supported source files via :class:`FileWalker`.
3. Chunk each file into semantic units via :class:`ASTChunker`.
4. Embed chunks in batches and upsert incrementally to ChromaDB.
5. Build (or reload) the BM25 keyword index via :class:`BM25Builder`.

Progress reporting
------------------
An optional ``progress_callback(phase, progress, label)`` coroutine is
called during chunking, embedding, and BM25 build so callers (e.g. the SSE
endpoint) can stream live progress updates.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Optional

from core.config import settings
from core.debug_log import log_timing
from ingestion.ast_chunker import ASTChunker
from ingestion.bm25_builder import BM25Builder
from ingestion.chroma_writer import ChromaWriter
from ingestion.embed_cache import (
    compute_content_hash,
    get_cached_vector,
    store_cached_vector,
)
from ingestion.embedding_service import EmbeddingService, _effective_batch_size
from ingestion.file_walker import FileWalker
from ingestion.git_cloner import GitCloner
from ingestion.manifest_builder import build_manifest_chunks
from ingestion.notebook_chunker import NotebookChunker
from core.dependencies import invalidate_bm25_cache
from core.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], Awaitable[None]]]

CHUNKING_WEIGHT = 0.20
EMBEDDING_WEIGHT = 0.70
BM25_WEIGHT = 0.10


def _with_embed_prefix(chunk: dict) -> dict:
    """Attach searchable embed text while preserving raw source text."""
    raw = chunk["text"]
    path = chunk.get("file_path", "")
    if chunk.get("chunk_type") == "file_manifest" or not path:
        chunk["embed_text"] = raw
    else:
        chunk["embed_text"] = f"File: {path}\n{raw}"
    chunk["content_hash"] = compute_content_hash(chunk["embed_text"])
    return chunk


def _chunk_file(source: str, rel_path: str, language: str) -> list[dict]:
    """Return semantic chunks for a single source file."""
    if language == "notebook":
        return NotebookChunker().chunk(source, rel_path)
    return ASTChunker(language).chunk(source, rel_path)


def _clear_existing_index(chroma_writer: ChromaWriter, collection_name: str) -> None:
    """Remove prior Chroma collection and BM25 cache before re-indexing."""
    try:
        chroma_writer.client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s' before re-index", collection_name)
    except Exception:
        pass

    cache_path = Path(settings.BM25_CACHE_DIR) / f"{collection_name}.json"
    if cache_path.is_file():
        cache_path.unlink()
        logger.info("Removed BM25 cache '%s'", cache_path)
    invalidate_bm25_cache(collection_name)


class IngestionOrchestrator:
    """Coordinates the full ingestion pipeline for a single repository."""

    def __init__(
        self,
        chroma_writer: ChromaWriter,
        embed_service: EmbeddingService,
    ) -> None:
        self._chroma = chroma_writer
        self._embed = embed_service
        self._cloner = GitCloner()
        self._walker = FileWalker()
        self._bm25 = BM25Builder()

    async def _embed_and_upsert_batch(
        self,
        chunks: list[dict],
        collection_name: str,
        repo_id: str,
    ) -> int:
        """Resolve embeddings for *chunks* and upsert to Chroma."""
        if not chunks:
            return 0

        vectors: list[list[float] | None] = [None] * len(chunks)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, chunk in enumerate(chunks):
            cached = get_cached_vector(chunk["content_hash"])
            if cached is not None:
                vectors[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(chunk.get("embed_text", chunk["text"]))

        if miss_texts:
            embed_t0 = time.perf_counter()
            new_vectors = await self._embed.embed_batch(miss_texts, task="document")
            for idx, vector in zip(miss_indices, new_vectors):
                vectors[idx] = vector
                store_cached_vector(chunks[idx]["content_hash"], vector)

            if settings.DEBUG_TIMING:
                embed_elapsed = time.perf_counter() - embed_t0
                cache_hits = len(chunks) - len(miss_texts)
                rate = len(miss_texts) / max(embed_elapsed, 1e-6)
                log_timing(
                    "ingest_embed_batch",
                    embed_elapsed * 1000,
                    {
                        "repo_id": repo_id,
                        "batch_chunks": len(chunks),
                        "misses": len(miss_texts),
                        "cache_hits": cache_hits,
                        "chunks_per_sec": round(rate, 2),
                    },
                )

        resolved = [v for v in vectors if v is not None]
        if len(resolved) != len(chunks):
            raise RuntimeError(
                f"Missing embeddings: expected {len(chunks)}, got {len(resolved)}"
            )

        batch_size = min(_effective_batch_size(), settings.CHROMA_UPSERT_BATCH_SIZE)
        loop = asyncio.get_event_loop()
        pending_upsert: asyncio.Future | None = None
        typed_vectors: list[list[float]] = [v for v in vectors if v is not None]

        for batch_start in range(0, len(chunks), batch_size):
            if pending_upsert is not None:
                await pending_upsert

            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_vectors = typed_vectors[batch_start:batch_end]

            pending_upsert = loop.run_in_executor(
                None,
                self._chroma.upsert,
                collection_name,
                batch_chunks,
                batch_vectors,
            )

        if pending_upsert is not None:
            await pending_upsert

        return len(chunks)

    async def ingest_repo(
        self,
        repo_url: str,
        branch: str = "main",
        progress_callback: ProgressCallback = None,
    ) -> dict:
        """Run the full ingestion pipeline for *repo_url*.

        Returns:
            ``{"repo_id": str, "collection": str, "chunks_indexed": int}``
        """
        repo_id = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        collection_name = f"repo_{repo_id}"
        flush_size = settings.INGEST_FLUSH_SIZE

        _clear_existing_index(self._chroma, collection_name)

        # ------------------------------------------------------------------
        # Step 1: Clone / pull
        # ------------------------------------------------------------------
        clone_path, _ = self._cloner.clone(repo_url, branch)

        # ------------------------------------------------------------------
        # Step 2: Walk files
        # ------------------------------------------------------------------
        files, skipped_paths = self._walker.walk_with_stats(clone_path)
        total_files = len(files)
        indexed_paths = [rel_path for rel_path, _, _ in files]
        skipped_summary = self._walker.summarize_skipped(skipped_paths)
        logger.info(
            "[%s] Ingesting %d files (%d assets skipped, branch=%s)",
            repo_id,
            total_files,
            len(skipped_paths),
            branch,
        )

        # ------------------------------------------------------------------
        # Steps 3–4: Chunk, embed, and upsert in bounded flush windows
        # ------------------------------------------------------------------
        buffer: list[dict] = []
        chunks_seen = 0
        total_indexed = 0
        chunked_paths: list[str] = []

        async def _report_embedding_progress() -> None:
            if progress_callback is None:
                return
            total = max(chunks_seen, total_indexed, 1)
            embed_fraction = total_indexed / total
            embed_progress = CHUNKING_WEIGHT + EMBEDDING_WEIGHT * embed_fraction
            await progress_callback(
                "embedding",
                embed_progress,
                f"Embedding chunks {total_indexed}/{total}",
            )

        async def _flush_buffer() -> None:
            nonlocal total_indexed
            while len(buffer) >= flush_size:
                batch = buffer[:flush_size]
                del buffer[:flush_size]
                total_indexed += await self._embed_and_upsert_batch(
                    batch, collection_name, repo_id
                )
                await _report_embedding_progress()

        for idx, (rel_path, full_path, language) in enumerate(files):
            try:
                source = open(  # noqa: WPS515
                    full_path, encoding="utf-8", errors="replace"
                ).read()
                chunks = _chunk_file(source, rel_path, language)
                chunks = [c for c in chunks if c.get("text", "").strip()]
                if chunks:
                    chunked_paths.append(rel_path)
                chunks = [_with_embed_prefix(c) for c in chunks]
                buffer.extend(chunks)
                chunks_seen += len(chunks)
            except Exception as exc:
                logger.warning("Skipping %s: %s", rel_path, exc)

            await _flush_buffer()

            if progress_callback is not None:
                chunk_progress = CHUNKING_WEIGHT * (idx + 1) / max(total_files, 1)
                await progress_callback(
                    "chunking",
                    chunk_progress,
                    f"Chunking files {idx + 1}/{total_files}",
                )

        if buffer:
            batch = buffer[:]
            buffer.clear()
            total_indexed += await self._embed_and_upsert_batch(
                batch, collection_name, repo_id
            )
            await _report_embedding_progress()

        manifest_chunks = build_manifest_chunks(
            chunked_paths,
            skipped_summary=skipped_summary,
        )
        manifest_chunks = [_with_embed_prefix(c) for c in manifest_chunks]
        chunks_seen += len(manifest_chunks)

        if manifest_chunks:
            total_indexed += await self._embed_and_upsert_batch(
                manifest_chunks, collection_name, repo_id
            )
            await _report_embedding_progress()

        buffer.clear()

        logger.info(
            "[%s] Total chunks indexed: %d (including %d manifest)",
            repo_id,
            total_indexed,
            len(manifest_chunks),
        )

        if total_indexed == 0:
            logger.warning("[%s] No chunks produced — aborting ingest", repo_id)
            return {
                "repo_id": repo_id,
                "collection": collection_name,
                "files_indexed": 0,
                "files_skipped": len(skipped_paths),
                "chunks_indexed": 0,
            }

        # ------------------------------------------------------------------
        # Step 5: Build / reload BM25 index
        # ------------------------------------------------------------------
        if progress_callback is not None:
            await progress_callback(
                "bm25",
                CHUNKING_WEIGHT + EMBEDDING_WEIGHT,
                "Building search index",
            )

        self._bm25.build_index(collection_name, self._chroma.client)

        from core.dependencies import get_bm25_data

        get_bm25_data(collection_name, self._chroma.client)

        if progress_callback is not None:
            await progress_callback("bm25", 1.0, "Indexing complete")

        logger.info(
            "[%s] Ingestion complete — %d chunks in '%s'",
            repo_id,
            total_indexed,
            collection_name,
        )
        return {
            "repo_id": repo_id,
            "collection": collection_name,
            "files_indexed": len(chunked_paths),
            "files_skipped": len(skipped_paths),
            "chunks_indexed": total_indexed,
        }
