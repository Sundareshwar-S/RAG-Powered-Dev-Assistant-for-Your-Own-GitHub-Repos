"""NDJSON debug logging for Cursor debug sessions."""
from __future__ import annotations

import json
import time
from pathlib import Path

_DEBUG_LOG = Path("/app/.cursor/debug-b99bf8.log")
if not _DEBUG_LOG.parent.exists():
    _alt = Path(__file__).resolve().parents[2] / ".cursor" / "debug-b99bf8.log"
    if _alt.parent.exists():
        _DEBUG_LOG = _alt
_SESSION_ID = "b99bf8"


def agent_debug_log(
    location: str,
    message: str,
    data: dict | None = None,
    *,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": _SESSION_ID,
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # endregion


def log_timing(stage: str, elapsed_ms: float, data: dict | None = None) -> None:
    """Record per-stage retrieval latency when DEBUG_TIMING is enabled."""
    from core.config import settings

    if not settings.DEBUG_TIMING:
        return
    agent_debug_log(
        f"timing:{stage}",
        f"{stage} completed in {elapsed_ms:.1f}ms",
        {"elapsed_ms": round(elapsed_ms, 2), **(data or {})},
        hypothesis_id="TIMING",
    )
