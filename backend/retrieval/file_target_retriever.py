"""Deterministic retrieval when a query names a specific indexed file."""
from __future__ import annotations

from ingestion.manifest_builder import dedupe_indexed_paths
from retrieval.query_intent import extract_target_file

MANIFEST_CHUNK_TYPE = "file_manifest"


def retrieve_file_target_context(
    query: str,
    corpus: list[dict],
) -> list[dict]:
    """Return all content chunks for the file named in *query*."""
    indexed_paths = dedupe_indexed_paths(corpus)
    target = extract_target_file(query, indexed_paths)
    if not target:
        return []

    target_lower = target.lower()
    matched = [
        chunk
        for chunk in corpus
        if chunk.get("chunk_type") != MANIFEST_CHUNK_TYPE
        and chunk.get("file_path", "").lower() == target_lower
    ]
    matched.sort(key=lambda c: int(c.get("start_line", 0)))
    return [{**chunk, "score": 1.0} for chunk in matched]
