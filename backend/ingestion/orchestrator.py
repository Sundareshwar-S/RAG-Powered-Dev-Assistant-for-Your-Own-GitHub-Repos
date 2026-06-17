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

import hashlib
from collections.abc import Awaitable, Callable
from typing import Optional

from core.config import settings
from ingestion.ast_chunker import ASTChunker
from ingestion.bm25_builder import BM25Builder
from ingestion.chroma_writer import ChromaWriter
from ingestion.embedding_service import EmbeddingService
from ingestion.file_walker import FileWalker
from ingestion.git_cloner import GitCloner
from core.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], Awaitable[None]]]

CHUNKING_WEIGHT = 0.20
EMBEDDING_WEIGHT = 0.70
BM25_WEIGHT = 0.10


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

        # ------------------------------------------------------------------
        # Step 1: Clone / pull
        # ------------------------------------------------------------------
        clone_path, _ = self._cloner.clone(repo_url, branch)

        # ------------------------------------------------------------------
        # Step 2: Walk files
        # ------------------------------------------------------------------
        files = self._walker.walk(clone_path)
        total_files = len(files)
        logger.info(
            "[%s] Ingesting %d files (branch=%s)", repo_id, total_files, branch
        )

        # ------------------------------------------------------------------
        # Step 3: Chunk all files
        # ------------------------------------------------------------------
        all_chunks: list[dict] = []

        for idx, (rel_path, full_path, language) in enumerate(files):
            try:
                source = open(  # noqa: WPS515
                    full_path, encoding="utf-8", errors="replace"
                ).read()
                chunks = ASTChunker(language).chunk(source, rel_path)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Skipping %s: %s", rel_path, exc)

            if progress_callback is not None:
                chunk_progress = CHUNKING_WEIGHT * (idx + 1) / max(total_files, 1)
                await progress_callback(
                    "chunking",
                    chunk_progress,
                    f"Chunking files {idx + 1}/{total_files}",
                )

        all_chunks = [c for c in all_chunks if c.get("text", "").strip()]

        logger.info("[%s] Total chunks produced: %d", repo_id, len(all_chunks))

        if not all_chunks:
            logger.warning("[%s] No chunks produced — aborting ingest", repo_id)
            return {
                "repo_id": repo_id,
                "collection": collection_name,
                "chunks_indexed": 0,
            }

        # ------------------------------------------------------------------
        # Step 4: Embed + upsert in pipelined batches
        # ------------------------------------------------------------------
        batch_size = settings.EMBED_LOCAL_BATCH_SIZE
        total_chunks = len(all_chunks)
        indexed = 0

        for batch_start in range(0, total_chunks, batch_size):
            batch_chunks = all_chunks[batch_start : batch_start + batch_size]
            texts = [c["text"] for c in batch_chunks]
            embeddings = await self._embed.embed_batch(texts)
            self._chroma.upsert(collection_name, batch_chunks, embeddings)
            indexed += len(batch_chunks)

            if progress_callback is not None:
                embed_fraction = indexed / total_chunks
                embed_progress = CHUNKING_WEIGHT + EMBEDDING_WEIGHT * embed_fraction
                await progress_callback(
                    "embedding",
                    embed_progress,
                    f"Embedding chunks {indexed}/{total_chunks}",
                )

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

        # Warm the in-memory BM25 cache so the first query avoids disk rebuild.
        from core.dependencies import get_bm25_data

        get_bm25_data(collection_name, self._chroma.client)

        if progress_callback is not None:
            await progress_callback("bm25", 1.0, "Indexing complete")

        logger.info(
            "[%s] Ingestion complete — %d chunks in '%s'",
            repo_id,
            indexed,
            collection_name,
        )
        return {
            "repo_id": repo_id,
            "collection": collection_name,
            "chunks_indexed": indexed,
        }
