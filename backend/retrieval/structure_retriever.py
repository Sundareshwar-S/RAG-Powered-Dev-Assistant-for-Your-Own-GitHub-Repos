"""Deterministic retrieval for file-structure and directory listing queries."""
from __future__ import annotations

from ingestion.manifest_builder import (
    build_folder_tree_text,
    dedupe_indexed_paths,
    filter_paths_by_prefix,
)
from retrieval.query_intent import extract_folder_prefix

MANIFEST_CHUNK_TYPE = "file_manifest"


def retrieve_structure_context(
    query: str,
    corpus: list[dict],
) -> list[dict]:
    """Return compact synthetic chunks with an authoritative file index."""
    manifest_chunks = [
        c for c in corpus if c.get("chunk_type") == MANIFEST_CHUNK_TYPE
    ]
    folder_prefix = extract_folder_prefix(query)
    indexed_paths = dedupe_indexed_paths(corpus)

    if manifest_chunks and not folder_prefix:
        return [{**c, "score": 1.0} for c in manifest_chunks]

    if folder_prefix:
        scoped = filter_paths_by_prefix(indexed_paths, folder_prefix)
        tree_text = build_folder_tree_text(indexed_paths, folder_prefix)
        header = (
            f"Indexed files under {folder_prefix}/ "
            f"({len(scoped)} file(s)):\n"
        )
    else:
        tree_text = build_folder_tree_text(indexed_paths)
        header = f"Indexed files in repository ({len(indexed_paths)} file(s)):\n"

    text = header + tree_text
    synthetic = {
        "text": text,
        "file_path": "__manifest__/filtered_tree.txt",
        "language": "plaintext",
        "chunk_type": MANIFEST_CHUNK_TYPE,
        "start_line": 1,
        "end_line": text.count("\n") + 1,
        "symbol_name": "",
        "score": 1.0,
    }
    return [synthetic]
