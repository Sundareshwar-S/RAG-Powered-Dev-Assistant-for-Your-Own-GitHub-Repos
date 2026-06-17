"""Async Ollama client for streaming LLM generation.

Protocol
--------
Ollama's ``/api/chat`` endpoint with ``stream: true`` returns a sequence of
newline-delimited JSON objects (NDJSON).  Each line is a JSON object with:

    {"model": "...", "message": {"role": "assistant", "content": "<token>"}, "done": false}

The final line has ``"done": true``.  We yield ``message.content`` from each
non-empty line and stop on ``done``.

Timeout
-------
A structured ``httpx.Timeout`` is used: 10s connect (fail fast when Ollama is
down), 300s read (allow slow CPU inference on the 7B model), 30s write, 10s
pool.  Using a uniform 300s timeout would cause 5-minute hangs on connection
failures.

Error handling
--------------
HTTP errors (e.g. 404 if a model is not pulled, 500 from Ollama) raise
``httpx.HTTPStatusError`` which propagates to the caller (API route or test).
Network errors raise ``httpx.RequestError``.  Neither is swallowed here —
the caller is responsible for turning them into structured API error responses.
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Use a structured timeout: short connect (fail fast if Ollama is down),
# long read (allow slow CPU inference to complete), generous write/pool.
_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=settings.OLLAMA_READ_TIMEOUT,
    write=30.0,
    pool=10.0,
)


async def stream_response(
    prompt_messages: list[dict],
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream LLM token output from Ollama.

    Args:
        prompt_messages: Ollama-compatible messages list, e.g. the output of
            ``prompt_builder.build_prompt()``.
        model: Ollama model tag (e.g. ``"qwen2.5-coder:7b"``).

    Yields:
        Individual token strings as they arrive from the model.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx status.
        httpx.RequestError: On network/connection failures.
    """
    payload = {
        "model": model,
        "messages": prompt_messages,
        "stream": True,
        "keep_alive": settings.OLLAMA_CHAT_KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "top_p": 0.9,
        },
    }

    url = f"{settings.OLLAMA_URL}/api/chat"
    logger.debug("Streaming from Ollama: model=%s url=%s", model, url)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line from Ollama: %r", raw_line)
                    continue

                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

                if data.get("done"):
                    logger.debug("Ollama stream complete (model=%s)", model)
                    return
