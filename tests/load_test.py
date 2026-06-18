"""Locust load test for the CodeBase Oracle API.

Simulates concurrent developer users querying an already-indexed repository.

Target (Phase 7 gate)
----------------------
- p95 query latency < 8 seconds (GPU-backed Ollama)

Usage
-----
# Interactive web UI (http://localhost:8089):
    locust -f tests/load_test.py --host http://localhost:8000

# Headless — 10 concurrent users, spawn rate 2/s, run for 60s:
    locust -f tests/load_test.py \
        --headless \
        --host http://localhost:8000 \
        -u 10 -r 2 --run-time 60s \
        --html tests/load_test_report.html

# Custom repo ID (default uses markupsafe md5 prefix):
    REPO_ID=abc12345 locust -f tests/load_test.py --headless ...

Environment variables
---------------------
REPO_ID:     8-char repo_id to query (default: md5 prefix of markupsafe URL).
ORACLE_MODEL: Ollama model to use (default: qwen2.5-coder:7b).

Prerequisites
-------------
- The stack must be running: docker compose up
- markupsafe (or another repo) must already be indexed
- Install locust: pip install locust
"""
from __future__ import annotations

import hashlib
import os
import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_REPO_ID = hashlib.md5(
    b"https://github.com/pallets/markupsafe", usedforsecurity=False
).hexdigest()[:8]

REPO_ID = os.environ.get("REPO_ID", _DEFAULT_REPO_ID)
ORACLE_MODEL = os.environ.get("ORACLE_MODEL", "qwen2.5-coder:7b")

# A realistic set of developer questions about code.
# These mirror questions a developer would ask against a typical Python library.
_QUERY_POOL = [
    "What does the escape function do?",
    "How does Markup prevent XSS attacks?",
    "What is the difference between escape and escape_silent?",
    "How does markupsafe handle None values?",
    "What is the Markup class and how does it inherit from str?",
    "How does string concatenation work with Markup objects?",
    "What characters are escaped by markupsafe?",
    "How does Markup.format escape its arguments?",
    "What is __html__ protocol and how does markupsafe use it?",
    "How does Markup.unescape work?",
    "What does striptags do?",
    "How is soft_str different from str?",
    "What is _MarkupEscapeHelper used for?",
    "How does markupsafe avoid double-escaping?",
    "What does the __mod__ method do on Markup?",
]


# ---------------------------------------------------------------------------
# Load test user
# ---------------------------------------------------------------------------


class OracleUser(HttpUser):
    """Simulates a developer querying the CodeBase Oracle.

    Task weights:
      - query (3): Most common action — heavy POST request that hits Ollama.
      - list_repos (1): Lightweight read.
      - health (1): Lightweight health probe.
    """

    wait_time = between(1, 3)

    @task(3)
    def query(self) -> None:
        """POST /api/v1/query — the hot path; exercises the full RAG pipeline."""
        question = random.choice(_QUERY_POOL)
        payload = {
            "repo_id": REPO_ID,
            "question": question,
            "model": ORACLE_MODEL,
        }
        with self.client.post(
            "/api/v1/query",
            json=payload,
            catch_response=True,
            name="POST /api/v1/query",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" not in data:
                    response.failure("Response missing 'answer' field")
                elif not data.get("answer"):
                    response.failure("Empty answer returned")
                else:
                    response.success()
            elif response.status_code == 404:
                response.failure(f"Repo {REPO_ID!r} not found — index it first")
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def list_repos(self) -> None:
        """GET /api/v1/repos — lightweight read to simulate UI polling."""
        with self.client.get(
            "/api/v1/repos",
            catch_response=True,
            name="GET /api/v1/repos",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if not isinstance(data, list):
                    response.failure("Expected list of repos")
                else:
                    response.success()
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def health(self) -> None:
        """GET /api/v1/health — lightweight probe to measure baseline latency."""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="GET /api/v1/health",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    response.success()
                else:
                    response.failure(f"Unhealthy: {data}")
            else:
                response.failure(f"Unexpected status {response.status_code}")
