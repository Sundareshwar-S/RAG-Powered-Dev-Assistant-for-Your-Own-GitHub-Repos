"""API contract tests for POST /query and WS /ws/chat."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _sample_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "text": f"def fn_{i}(): pass",
            "file_path": f"src/mod_{i}.py",
            "language": "python",
            "chunk_type": "function",
            "start_line": i * 10,
            "end_line": i * 10 + 5,
            "symbol_name": f"fn_{i}",
            "score": 1.0,
        }
        for i in range(n)
    ]


async def _async_gen(*tokens):
    for t in tokens:
        yield t


class _WSRecorder:
    """Minimal WebSocket stub that records outbound messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        return json.dumps({"repo_id": "abc", "question": self._question})

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000) -> None:
        self.closed = True


@pytest.mark.asyncio
class TestWSChat:
    async def test_conversational_short_circuit(self, mock_embed_service, tmp_chroma):
        from api.routes_query import ws_chat
        from core.config import settings

        ws = _WSRecorder()
        ws._question = "hi"

        await ws_chat(ws, mock_embed_service, tmp_chroma, settings)

        types = [m["type"] for m in ws.sent]
        assert types == ["sources", "token", "done"]
        assert "CodeBase Oracle" in ws.sent[1]["token"]

    async def test_happy_path_message_sequence(self, mock_embed_service, tmp_chroma):
        from api.routes_query import ws_chat
        from core.config import settings

        chunks = _sample_chunks(13)
        mock_retriever = MagicMock()
        mock_retriever._collection.count.return_value = 13
        mock_retriever.retrieve = AsyncMock(return_value=chunks)

        ws = _WSRecorder()
        ws._question = "what's in the repo"

        with (
            patch("api.routes_query._build_retriever", AsyncMock(return_value=mock_retriever)),
            patch(
                "api.routes_query.ollama_client.stream_response",
                return_value=_async_gen("The repo contains "),
            ),
        ):
            await ws_chat(ws, mock_embed_service, tmp_chroma, settings)

        types = [m["type"] for m in ws.sent]
        assert types[0] == "status"
        assert ws.sent[0]["phase"] == "loading_index"
        assert "sources" in types
        assert "generating" in [m.get("phase") for m in ws.sent if m.get("type") == "status"]
        assert "token" in types
        assert types[-1] == "done"

    async def test_retrieval_error_sends_error_type(self, mock_embed_service, tmp_chroma):
        from api.routes_query import ws_chat
        from core.config import settings

        ws = _WSRecorder()
        ws._question = "explain foo"

        with patch(
            "api.routes_query._build_retriever",
            AsyncMock(side_effect=RuntimeError("BM25 missing")),
        ):
            await ws_chat(ws, mock_embed_service, tmp_chroma, settings)

        assert ws.sent[-1]["type"] == "error"
        assert "BM25 missing" in ws.sent[-1]["message"]

    async def test_guardrail_sends_no_context_token(self, mock_embed_service, tmp_chroma):
        from api.routes_query import ws_chat
        from core.config import settings

        chunks = [{
            "text": "x",
            "file_path": "a.py",
            "score": 0.01,
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "fn",
            "symbol_name": "",
        }]
        mock_retriever = MagicMock()
        mock_retriever._collection.count.return_value = 100
        mock_retriever.retrieve = AsyncMock(return_value=chunks)

        ws = _WSRecorder()
        ws._question = "explain quantum physics"

        with patch("api.routes_query._build_retriever", AsyncMock(return_value=mock_retriever)):
            await ws_chat(ws, mock_embed_service, tmp_chroma, settings)

        token_msgs = [m for m in ws.sent if m.get("type") == "token"]
        assert token_msgs
        assert "don't have enough context" in token_msgs[0]["token"].lower()


@pytest.mark.asyncio
class TestPostQuery:
    async def test_conversational_skips_retrieval(self, mock_embed_service, tmp_chroma):
        from starlette.requests import Request

        from api.routes_query import query
        from core.config import settings
        from models.schemas import QueryRequest

        scope = {"type": "http", "method": "POST", "path": "/api/v1/query", "headers": []}
        request = Request(scope)
        body = QueryRequest(repo_id="abc", question="hello")

        result = await query(request, body, mock_embed_service, tmp_chroma, settings)

        assert "CodeBase Oracle" in result.answer
        assert result.sources == []

    async def test_query_happy_path(self, mock_embed_service, tmp_chroma):
        from starlette.requests import Request

        from api.routes_query import query
        from core.config import settings
        from models.schemas import QueryRequest

        chunks = _sample_chunks(2)
        mock_retriever = MagicMock()
        mock_retriever._collection.count.return_value = 2
        mock_retriever.retrieve = AsyncMock(return_value=chunks)

        scope = {"type": "http", "method": "POST", "path": "/api/v1/query", "headers": []}
        request = Request(scope)
        body = QueryRequest(repo_id="abc", question="what does fn_0 do?")

        with (
            patch("api.routes_query._build_retriever", AsyncMock(return_value=mock_retriever)),
            patch(
                "api.routes_query.ollama_client.stream_response",
                return_value=_async_gen("answer text"),
            ),
        ):
            result = await query(request, body, mock_embed_service, tmp_chroma, settings)

        assert result.answer == "answer text"
        assert len(result.sources) == 2
