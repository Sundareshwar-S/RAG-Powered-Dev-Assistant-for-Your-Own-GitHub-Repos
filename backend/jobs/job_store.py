"""In-memory job store for background ingestion tasks.

Thread-safety
-------------
All mutations are protected by ``_lock`` so that the FastAPI background task
(runs in a thread pool) and the SSE polling endpoint (runs in the asyncio event
loop thread) can safely read/write concurrently without data races.

Job schema
----------
Each job is stored as a plain dict:

    {
        "job_id":       str,          # UUID
        "repo_id":      str,          # md5(repo_url)[:8]
        "status":       str,          # "running" | "completed" | "failed"
        "progress":     float,        # 0.0 – 1.0
        "phase":        str,          # "chunking" | "embedding" | "bm25" | ""
        "current_file": str,          # human-readable progress label
        "error":        str | None,   # set on failure
    }
"""
from __future__ import annotations

import threading
from typing import Optional

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(job_id: str, repo_id: str) -> dict:
    """Create a new job record and return it."""
    record: dict = {
        "job_id": job_id,
        "repo_id": repo_id,
        "status": "running",
        "progress": 0.0,
        "phase": "",
        "current_file": "",
        "error": None,
    }
    with _lock:
        _jobs[job_id] = record
    return dict(record)


def update_job(
    job_id: str,
    progress: float,
    current_file: str = "",
    phase: str = "",
) -> None:
    """Update progress, phase, and label for a running job."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["progress"] = float(progress)
            job["current_file"] = current_file
            if phase:
                job["phase"] = phase


def complete_job(job_id: str) -> None:
    """Mark job as completed at 100 % progress."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "completed"
            job["progress"] = 1.0
            job["phase"] = "completed"
            job["current_file"] = ""


def fail_job(job_id: str, error: str) -> None:
    """Mark job as failed with the given error message."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "failed"
            job["error"] = error


def get_job(job_id: str) -> Optional[dict]:
    """Return the job dict for *job_id*, or ``None`` if not found."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def get_active_job_for_repo(repo_id: str) -> Optional[str]:
    """Return the job_id of the currently running job for *repo_id*, or ``None``.

    Checks only jobs with ``status == "running"``; completed/failed jobs are
    not considered active and do not block a new ingest.
    """
    with _lock:
        for job_id, job in _jobs.items():
            if job["repo_id"] == repo_id and job["status"] == "running":
                return job_id
    return None
