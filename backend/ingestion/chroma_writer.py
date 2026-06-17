"""ChromaDB writer for code chunks.

Version note (ChromaDB 0.5.23):
- ``PersistentClient`` is used for file-based persistence.
- ``client.list_collections()`` returns ``Collection`` objects with a
  ``.name`` attribute — NOT plain strings.  This changed in 0.6.x.
- Collections are created/retrieved via ``get_or_create_collection``
  with cosine distance so that query scores are interpretable as
  similarity values.
"""
from __future__ import annotations

import uuid
from typing import Any

import chromadb

from core.logger import get_logger

logger = get_logger(__name__)


class ChromaWriter:
    """Writes code chunks (text + embeddings + metadata) into ChromaDB."""

    def __init__(self, chroma_path: str = "/chroma_db") -> None:
        self.client = chromadb.PersistentClient(path=chroma_path)

    def upsert(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> chromadb.Collection:
        """Upsert *chunks* with their *embeddings* into *collection_name*.

        Each chunk must have the keys produced by :class:`ASTChunker`:
        ``text``, ``file_path``, ``language``, ``chunk_type``,
        ``start_line``, ``end_line``, ``symbol_name``.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "file_path": c["file_path"],
                "language": c["language"],
                "chunk_type": c["chunk_type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "symbol_name": c["symbol_name"],
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Upserted %d chunks → collection '%s' (total: %d)",
            len(chunks),
            collection_name,
            collection.count(),
        )
        return collection

    def list_collection_names(self) -> list[str]:
        """Return collection names.  Compatible with ChromaDB 0.5.x API."""
        # 0.5.x: list_collections() returns Collection objects with .name
        return [col.name for col in self.client.list_collections()]
