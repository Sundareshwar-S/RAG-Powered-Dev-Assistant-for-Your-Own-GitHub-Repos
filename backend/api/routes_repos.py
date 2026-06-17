"""Repository management and health-check routes.

Endpoints
---------
GET /repos
    List all indexed repositories.

DELETE /repos/{repo_id}
    Delete a repository's ChromaDB collection and BM25 cache file.

GET /health
    Check connectivity to Ollama and ChromaDB.
"""
from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path as PathParam

from core.dependencies import get_chroma_client, get_settings, invalidate_bm25_cache
from core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["repos"])


# ---------------------------------------------------------------------------
# GET /repos
# ---------------------------------------------------------------------------


@router.get("/repos")
async def list_repos(
    chroma_client=Depends(get_chroma_client),
) -> list[dict]:
    """List all indexed repositories.

    Returns a list of dicts with ``repo_id``, ``collection``, and ``chunks``.
    ChromaDB 0.5.x does not persist repo URL or indexed_at timestamp in
    collection metadata, so those fields default to empty/null.
    """
    collections = chroma_client.list_collections()
    result = []
    for col in collections:
        name: str = col.name  # ChromaDB 0.5.x: .name attribute
        if not name.startswith("repo_"):
            continue

        repo_id = name.removeprefix("repo_")
        try:
            collection = chroma_client.get_collection(name)
            chunks = collection.count()
        except Exception as exc:
            logger.warning("Could not count chunks for %s: %s", name, exc)
            chunks = -1

        result.append(
            {
                "repo_id": repo_id,
                "collection": name,
                "chunks": chunks,
                "indexed_at": None,
            }
        )

    return result


# ---------------------------------------------------------------------------
# DELETE /repos/{repo_id}
# ---------------------------------------------------------------------------


@router.delete("/repos/{repo_id}")
async def delete_repo(
    repo_id: str = PathParam(..., pattern=r"^[a-f0-9]{8}$"),
    chroma_client=Depends(get_chroma_client),
    settings=Depends(get_settings),
) -> dict:
    """Delete the ChromaDB collection and BM25 cache file for *repo_id*.

    Returns HTTP 404 if the collection does not exist.
    """
    collection_name = f"repo_{repo_id}"

    # Check existence first
    existing = [col.name for col in chroma_client.list_collections()]
    if collection_name not in existing:
        raise HTTPException(status_code=404, detail=f"Repo {repo_id!r} not found")

    # Delete ChromaDB collection
    try:
        chroma_client.delete_collection(collection_name)
        logger.info("Deleted ChromaDB collection '%s'", collection_name)
    except Exception as exc:
        logger.exception("Failed to delete collection '%s': %s", collection_name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {exc}") from exc

    # Delete BM25 cache file
    bm25_path = Path(settings.BM25_CACHE_DIR) / f"{collection_name}.json"
    if bm25_path.is_file():
        try:
            bm25_path.unlink()
            logger.info("Deleted BM25 cache file: %s", bm25_path)
        except OSError as exc:
            logger.warning("Could not delete BM25 cache file %s: %s", bm25_path, exc)

    # Evict from in-memory BM25 cache
    invalidate_bm25_cache(collection_name)

    return {"deleted": True, "repo_id": repo_id}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(
    chroma_client=Depends(get_chroma_client),
    settings=Depends(get_settings),
) -> dict:
    """Return health status for all downstream services."""
    # --- Ollama ---
    ollama_status = "ok"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        ollama_status = f"error: {exc}"

    # --- ChromaDB ---
    chroma_status = "ok"
    try:
        chroma_client.list_collections()
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)
        chroma_status = f"error: {exc}"

    overall = "ok" if ollama_status == "ok" and chroma_status == "ok" else "degraded"

    return {
        "status": overall,
        "ollama": ollama_status,
        "chroma": chroma_status,
    }
