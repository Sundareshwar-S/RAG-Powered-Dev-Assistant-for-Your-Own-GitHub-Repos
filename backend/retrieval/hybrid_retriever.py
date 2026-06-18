"""Hybrid retriever: dense + sparse + RRF + cross-encoder reranking.

For structure queries, returns a compact authoritative file index.
For small corpora (<= SMALL_CORPUS_THRESHOLD), returns all indexed chunks
without Ollama embed or cross-encoder.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import chromadb
import chromadb.api
from rank_bm25 import BM25Okapi

from core.config import settings
from core.debug_log import log_timing
from core.logger import get_logger
from ingestion.embedding_service import EmbeddingService
from retrieval.dense_retriever import DenseRetriever
from ingestion.manifest_builder import dedupe_indexed_paths
from retrieval.file_target_retriever import retrieve_file_target_context
from retrieval.query_intent import extract_target_file, is_structure_query
from retrieval.reranker import Reranker
from retrieval.rrf_fusion import merge as rrf_merge
from retrieval.sparse_retriever import SparseRetriever
from retrieval.structure_retriever import retrieve_structure_context

logger = get_logger(__name__)

_DENSE_K = 50
_SPARSE_K = 50
_FINAL_K = 8

StatusCallback = Callable[[str], Awaitable[None]] | None


def sort_corpus_for_overview(corpus: list[dict]) -> list[dict]:
    """Sort chunks for overview answers: README first, then path and line."""

    def _sort_key(chunk: dict) -> tuple:
        path = chunk.get("file_path", "").lower()
        readme_rank = 0 if "readme" in path else 1
        return (readme_rank, path, int(chunk.get("start_line", 0)))

    return sorted(corpus, key=_sort_key)


class HybridRetriever:
    """Orchestrates the full multi-stage retrieval pipeline."""

    def __init__(
        self,
        collection_name: str,
        chroma_client: chromadb.api.ClientAPI,
        embed_service: EmbeddingService,
        bm25_index: BM25Okapi,
        corpus: list[dict],
    ) -> None:
        self._collection_name = collection_name
        self._collection = chroma_client.get_collection(collection_name)
        self._embed_service = embed_service
        self._bm25_index = bm25_index
        self._corpus = corpus

        self._dense = DenseRetriever()
        self._sparse = SparseRetriever()
        self._reranker = Reranker()

    async def retrieve(
        self,
        query: str,
        final_k: int = _FINAL_K,
        status_callback: StatusCallback = None,
    ) -> list[dict]:
        """Run retrieval for *query*, using a fast path for tiny corpora."""
        logger.info(
            "HybridRetriever: query=%r collection=%s",
            query[:80],
            self._collection_name,
        )

        corpus_size = self._collection.count()

        indexed_paths = dedupe_indexed_paths(self._corpus)
        if extract_target_file(query, indexed_paths):
            return await self._retrieve_file_target_path(
                query, corpus_size, status_callback
            )

        if is_structure_query(query):
            return await self._retrieve_structure_path(
                query, corpus_size, status_callback
            )

        if corpus_size <= settings.SMALL_CORPUS_THRESHOLD:
            return await self._retrieve_fast_path(query, corpus_size, status_callback)

        return await self._retrieve_full_pipeline(
            query, final_k, corpus_size, status_callback
        )

    async def _retrieve_file_target_path(
        self,
        query: str,
        corpus_size: int,
        status_callback: StatusCallback,
    ) -> list[dict]:
        """Return all chunks for a file explicitly named in the query."""
        t0 = time.perf_counter()
        if status_callback:
            await status_callback("ranking")

        results = retrieve_file_target_context(query, self._corpus)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_timing(
            "file_target_path",
            elapsed_ms,
            {
                "query_prefix": query[:80],
                "corpus_size": corpus_size,
                "final": len(results),
            },
        )
        logger.info(
            "HybridRetriever file-target path: corpus=%d final=%d (%.1fms)",
            corpus_size,
            len(results),
            elapsed_ms,
        )
        return results

    async def _retrieve_structure_path(
        self,
        query: str,
        corpus_size: int,
        status_callback: StatusCallback,
    ) -> list[dict]:
        """Return compact file-index chunks for structure/listing questions."""
        t0 = time.perf_counter()
        if status_callback:
            await status_callback("ranking")

        results = retrieve_structure_context(query, self._corpus)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_timing(
            "structure_path",
            elapsed_ms,
            {
                "query_prefix": query[:80],
                "corpus_size": corpus_size,
                "final": len(results),
            },
        )
        logger.info(
            "HybridRetriever structure path: corpus=%d final=%d (%.1fms)",
            corpus_size,
            len(results),
            elapsed_ms,
        )
        return results

    async def _retrieve_fast_path(
        self,
        query: str,
        corpus_size: int,
        status_callback: StatusCallback,
    ) -> list[dict]:
        """Return all corpus chunks without embed/rerank (small repos / overview)."""
        t0 = time.perf_counter()
        if status_callback:
            await status_callback("ranking")

        sorted_chunks = sort_corpus_for_overview(self._corpus)
        results = [{**chunk, "score": 1.0} for chunk in sorted_chunks]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_timing(
            "fast_path",
            elapsed_ms,
            {"query_prefix": query[:80], "corpus_size": corpus_size, "final": len(results)},
        )
        logger.info(
            "HybridRetriever fast path: corpus=%d final=%d (%.1fms)",
            corpus_size,
            len(results),
            elapsed_ms,
        )
        return results

    async def _retrieve_full_pipeline(
        self,
        query: str,
        final_k: int,
        corpus_size: int,
        status_callback: StatusCallback,
    ) -> list[dict]:
        """Full dense + sparse + RRF + cross-encoder pipeline for large repos."""
        loop = asyncio.get_event_loop()
        timings: dict[str, float] = {}

        if status_callback:
            await status_callback("embedding")

        # Sparse does not need the embedding — run in parallel with embed+dense.
        t_sparse = time.perf_counter()
        sparse_task = loop.run_in_executor(
            None,
            self._sparse.retrieve,
            query,
            self._bm25_index,
            self._corpus,
            _SPARSE_K,
        )

        t_embed = time.perf_counter()
        query_embeddings = await self._embed_service.embed_batch(
            [query], keep_alive="0", task="query"
        )
        timings["embed_ms"] = (time.perf_counter() - t_embed) * 1000

        if not query_embeddings:
            logger.error(
                "EmbeddingService returned empty result for query: %r", query[:80]
            )
            await sparse_task
            return []
        query_embedding = query_embeddings[0]

        t_dense = time.perf_counter()
        dense_results = await loop.run_in_executor(
            None,
            self._dense.retrieve,
            query_embedding,
            self._collection,
            _DENSE_K,
        )
        timings["dense_ms"] = (time.perf_counter() - t_dense) * 1000

        sparse_results = await sparse_task
        timings["sparse_ms"] = (time.perf_counter() - t_sparse) * 1000

        t_rrf = time.perf_counter()
        fused = rrf_merge(dense_results, sparse_results)
        timings["rrf_ms"] = (time.perf_counter() - t_rrf) * 1000

        if not fused:
            logger.warning("RRF fusion returned no results for query: %r", query[:80])
            return []

        dense_by_text = {c["text"][:100]: c.get("score", 0.0) for c in dense_results}
        for chunk in fused:
            key = chunk["text"][:100]
            chunk["dense_score"] = dense_by_text.get(key, 0.0)

        if len(fused) <= final_k:
            for chunk in fused:
                chunk["score"] = chunk.get("dense_score", chunk.get("rrf_score", 0.0))
            fused.sort(key=lambda c: c.get("score", 0.0), reverse=True)
            final = fused[:final_k]
        else:
            if status_callback:
                await status_callback("reranking")
            t_rerank = time.perf_counter()
            final = await loop.run_in_executor(
                None, self._reranker.rerank, query, fused, final_k
            )
            timings["rerank_ms"] = (time.perf_counter() - t_rerank) * 1000

        total_ms = sum(timings.values())
        log_timing(
            "full_pipeline",
            total_ms,
            {
                "query_prefix": query[:80],
                "corpus_size": corpus_size,
                **{k: round(v, 2) for k, v in timings.items()},
                "dense": len(dense_results),
                "sparse": len(sparse_results),
                "fused": len(fused),
                "final": len(final),
            },
        )
        logger.info(
            "HybridRetriever: dense=%d sparse=%d fused=%d final=%d (%.0fms)",
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(final),
            total_ms,
        )
        return final
