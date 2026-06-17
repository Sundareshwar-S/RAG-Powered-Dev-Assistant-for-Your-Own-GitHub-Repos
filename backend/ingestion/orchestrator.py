"""Ingestion orchestrator — ties together all Phase 1 pipeline steps.

Pipeline
--------
1. Clone / pull the repo via :class:`GitCloner`.
2. Walk all supported source files via :class:`FileWalker`.
3. Chunk each file into semantic units via :class:`ASTChunker`.
4. Embed all chunks in batches via :class:`EmbeddingService`.
5. Write chunks + embeddings to ChromaDB via :class:`ChromaWriter`.
6. Build (or reload) the BM25 keyword index via :class:`BM25Builder`.

Progress reporting
------------------
An optional ``progress_callback(current: int, total: int)`` coroutine is
called after each file is processed so callers (e.g. the SSE endpoint) can
stream live progress updates.
"""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Optional

from ingestion.ast_chunker import ASTChunker
from ingestion.bm25_builder import BM25Builder
from ingestion.chroma_writer import ChromaWriter
from ingestion.embedding_service import EmbeddingService
from ingestion.file_walker import FileWalker
from ingestion.git_cloner import GitCloner
from core.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[int, int], Awaitable[None]]]


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
                await progress_callback(idx + 1, total_files)

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
        # Step 4: Embed all chunks
        # ------------------------------------------------------------------
        texts = [c["text"] for c in all_chunks]
        embeddings = await self._embed.embed_batch(texts)

        # ------------------------------------------------------------------
        # Step 5: Write to ChromaDB
        # ------------------------------------------------------------------
        self._chroma.upsert(collection_name, all_chunks, embeddings)

        # ------------------------------------------------------------------
        # Step 6: Build / reload BM25 index
        # ------------------------------------------------------------------
        self._bm25.build_index(collection_name, self._chroma.client)

        # Warm the in-memory BM25 cache so the first query avoids disk rebuild.
        from core.dependencies import get_bm25_data

        get_bm25_data(collection_name, self._chroma.client)

        logger.info(
            "[%s] Ingestion complete — %d chunks in '%s'",
            repo_id,
            len(all_chunks),
            collection_name,
        )
        return {
            "repo_id": repo_id,
            "collection": collection_name,
            "chunks_indexed": len(all_chunks),
        }
