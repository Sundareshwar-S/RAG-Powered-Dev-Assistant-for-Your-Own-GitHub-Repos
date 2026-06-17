"""Ingestion API routes.

Endpoints
---------
POST /ingest
    Kicks off a background ingestion job for a GitHub repository.
    Returns HTTP 409 if the same repo is already being ingested.

GET /ingest/{job_id}/status
    Server-Sent Events (SSE) stream that pushes progress updates every
    second until the job completes or fails.
"""
import asyncio
import hashlib
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

import jobs.job_store as job_store
from core.debug_log import agent_debug_log
from core.dependencies import get_chroma_client, get_embed_service
from core.limiter import limiter
from core.logger import get_logger
from ingestion.chroma_writer import ChromaWriter
from ingestion.embedding_service import EmbeddingService
from ingestion.orchestrator import IngestionOrchestrator
from models.schemas import IngestRequest

logger = get_logger(__name__)
router = APIRouter(tags=["ingestion"])


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", status_code=202)
@limiter.limit("5/minute")
async def start_ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    embed_service: EmbeddingService = Depends(get_embed_service),
    chroma_client=Depends(get_chroma_client),
) -> dict:
    """Start a background ingestion job for *repo_url*.

    Returns ``{"job_id": str, "status": "running"}`` on success, or HTTP 409
    if the same repo is currently being ingested.
    """
    repo_url = str(body.repo_url).strip().rstrip("/")
    if repo_url.lower().endswith(".git"):
        repo_url = repo_url[:-4]
    repo_id = hashlib.md5(repo_url.encode(), usedforsecurity=False).hexdigest()[:8]

    agent_debug_log(
        "routes_ingest.py:start_ingest",
        "Ingest request accepted",
        {"repo_url": repo_url, "branch": body.branch, "repo_id": repo_id},
        hypothesis_id="H5",
    )

    # Duplicate ingest guard
    existing_job_id = job_store.get_active_job_for_repo(repo_id)
    if existing_job_id is not None:
        logger.warning(
            "Repo %s is already being ingested (job_id=%s)", repo_id, existing_job_id
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "This repo is already being ingested",
                "job_id": existing_job_id,
            },
        )

    job_id = str(uuid.uuid4())
    job_store.create_job(job_id, repo_id)

    logger.info("Starting ingest job %s for repo %s (branch=%s)", job_id, repo_id, body.branch)

    background_tasks.add_task(
        _run_ingest,
        job_id=job_id,
        repo_url=repo_url,
        branch=body.branch,
        embed_service=embed_service,
        chroma_client=chroma_client,
    )

    return {"job_id": job_id, "status": "running"}


async def _run_ingest(
    job_id: str,
    repo_url: str,
    branch: str,
    embed_service: EmbeddingService,
    chroma_client,
) -> None:
    """Background task that runs the full ingestion pipeline."""
    try:
        chroma_writer = ChromaWriter()
        orchestrator = IngestionOrchestrator(
            chroma_writer=chroma_writer,
            embed_service=embed_service,
        )

        async def _progress(phase: str, progress: float, label: str) -> None:
            job_store.update_job(
                job_id,
                progress=progress,
                current_file=label,
                phase=phase,
            )

        result = await orchestrator.ingest_repo(
            repo_url=repo_url,
            branch=branch,
            progress_callback=_progress,
        )

        logger.info(
            "Ingest job %s completed: %d chunks indexed",
            job_id,
            result.get("chunks_indexed", 0),
        )
        job_store.complete_job(job_id)

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        logger.exception("Ingest job %s failed: %s", job_id, error_msg)
        agent_debug_log(
            "routes_ingest.py:_run_ingest",
            "Ingest job failed",
            {"job_id": job_id, "repo_url": repo_url, "error": error_msg},
            hypothesis_id="H5",
        )
        job_store.fail_job(job_id, error_msg)


# ---------------------------------------------------------------------------
# GET /ingest/{job_id}/status — SSE stream
# ---------------------------------------------------------------------------


@router.get("/ingest/{job_id}/status")
async def ingest_status(job_id: str) -> StreamingResponse:
    """Stream SSE progress events for *job_id*.

    Yields ``data: <json>\\n\\n`` events every second until the job is in a
    terminal state (``completed`` or ``failed``), then sends a final event
    and closes the stream.
    """
    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator(job_id: str) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted JSON lines."""
    while True:
        job = job_store.get_job(job_id)

        if job is None:
            # Unknown job — send error event and close
            payload = json.dumps({"status": "not_found", "job_id": job_id})
            yield f"data: {payload}\n\n"
            return

        payload = json.dumps(
            {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "phase": job.get("phase", ""),
                "current_file": job["current_file"],
                "error": job["error"],
            }
        )
        yield f"data: {payload}\n\n"

        if job["status"] in ("completed", "failed"):
            return

        await asyncio.sleep(1.0)
