"""Query and streaming chat API routes."""
import asyncio
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import retrieval.ollama_client as ollama_client
from core.debug_log import agent_debug_log, log_timing
from core.dependencies import get_bm25_data, get_chroma_client, get_embed_service, get_settings
from core.limiter import limiter
from core.logger import get_logger
from ingestion.embedding_service import EmbeddingService
from models.schemas import QueryRequest, QueryResponse, SourceChunk
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.prompt_builder import (
    build_prompt,
    get_conversational_response,
    has_sufficient_context,
    is_conversational_query,
)

logger = get_logger(__name__)
router = APIRouter(tags=["query"])

_NO_CONTEXT_RESPONSE = (
    "I don't have enough context in the indexed codebase to answer this confidently."
)


class WSChatRequest(BaseModel):
    """Validated WebSocket chat message payload."""
    repo_id: str
    question: str
    model: str | None = None


def _collection_name(repo_id: str) -> str:
    return f"repo_{repo_id}"


def _chunks_to_sources(chunks: list[dict]) -> list[SourceChunk]:
    return [
        SourceChunk(
            file_path=c.get("file_path", ""),
            start_line=int(c.get("start_line", 0)),
            end_line=int(c.get("end_line", 0)),
            chunk_type=c.get("chunk_type", "snippet"),
            symbol_name=c.get("symbol_name", ""),
            text=c.get("text", ""),
        )
        for c in chunks
    ]


async def _build_retriever(
    repo_id: str,
    embed_service: EmbeddingService,
    chroma_client,
) -> HybridRetriever:
    """Build a HybridRetriever for the given repo."""
    collection_name = _collection_name(repo_id)
    loop = asyncio.get_event_loop()
    bm25_index, corpus = await loop.run_in_executor(
        None, get_bm25_data, collection_name, chroma_client
    )
    return HybridRetriever(
        collection_name=collection_name,
        chroma_client=chroma_client,
        embed_service=embed_service,
        bm25_index=bm25_index,
        corpus=corpus,
    )


async def _retrieve_with_timeout(
    retriever: HybridRetriever,
    question: str,
    timeout: float,
    status_callback=None,
) -> list[dict]:
    """Run retrieval with a wall-clock timeout."""
    return await asyncio.wait_for(
        retriever.retrieve(question, status_callback=status_callback),
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# POST /query — non-streaming
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query(
    request: Request,
    body: QueryRequest,
    embed_service: EmbeddingService = Depends(get_embed_service),
    chroma_client=Depends(get_chroma_client),
    settings=Depends(get_settings),
) -> QueryResponse:
    """Run a full RAG query and return the answer with source citations."""
    model = body.model or settings.DEFAULT_LLM_MODEL
    logger.info(
        "Query request: repo_id=%s model=%s question=%r",
        body.repo_id,
        model,
        body.question[:80],
    )

    if is_conversational_query(body.question):
        return QueryResponse(
            answer=get_conversational_response(),
            sources=[],
            model_used=model,
        )

    t0 = time.perf_counter()
    try:
        retriever = await _build_retriever(body.repo_id, embed_service, chroma_client)
        corpus_size = retriever._collection.count()  # noqa: SLF001
        chunks = await _retrieve_with_timeout(
            retriever,
            body.question,
            settings.RETRIEVAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Retrieval timed out for repo_id=%s", body.repo_id)
        raise HTTPException(
            status_code=504,
            detail=f"Retrieval timed out after {settings.RETRIEVAL_TIMEOUT}s",
        ) from None
    except Exception as exc:
        logger.exception("Retrieval failed for repo_id=%s: %s", body.repo_id, exc)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    retrieval_ms = (time.perf_counter() - t0) * 1000
    log_timing(
        "routes_query.retrieve",
        retrieval_ms,
        {"repo_id": body.repo_id, "chunk_count": len(chunks)},
    )

    sufficient = has_sufficient_context(chunks, corpus_size=corpus_size)
    agent_debug_log(
        "routes_query.py:query",
        "Guardrail decision",
        {
            "repo_id": body.repo_id,
            "question_prefix": body.question[:80],
            "chunk_count": len(chunks),
            "corpus_size": corpus_size,
            "max_score": max((c.get("score", 0.0) for c in chunks), default=0.0),
            "sufficient": sufficient,
            "retrieval_ms": round(retrieval_ms, 2),
        },
        hypothesis_id="H1",
    )
    if not sufficient:
        logger.info(
            "Guardrail triggered — insufficient context for question: %r", body.question[:80]
        )
        return QueryResponse(
            answer=_NO_CONTEXT_RESPONSE,
            sources=[],
            model_used=model,
        )

    messages = build_prompt(body.question, chunks)

    try:
        answer_parts: list[str] = []
        async for token in ollama_client.stream_response(messages, model):
            answer_parts.append(token)
    except Exception as exc:
        logger.exception("Ollama streaming failed for repo_id=%s: %s", body.repo_id, exc)
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    return QueryResponse(
        answer="".join(answer_parts),
        sources=_chunks_to_sources(chunks),
        model_used=model,
    )


# ---------------------------------------------------------------------------
# WS /ws/chat — streaming WebSocket
# ---------------------------------------------------------------------------


@router.websocket("/ws/chat")
async def ws_chat(
    websocket: WebSocket,
    embed_service: EmbeddingService = Depends(get_embed_service),
    chroma_client=Depends(get_chroma_client),
    settings=Depends(get_settings),
) -> None:
    """WebSocket streaming chat endpoint."""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        chat_req = WSChatRequest(**payload)
    except WebSocketDisconnect:
        return
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("WS: failed to parse initial message: %s", exc)
        await websocket.close(code=1003)
        return

    model = chat_req.model or settings.DEFAULT_LLM_MODEL

    if not chat_req.repo_id or not chat_req.question.strip():
        await websocket.send_text(
            json.dumps({"type": "error", "message": "repo_id and question are required"})
        )
        await websocket.close(code=1003)
        return

    logger.info(
        "WS chat: repo_id=%s model=%s question=%r",
        chat_req.repo_id,
        model,
        chat_req.question[:80],
    )

    if is_conversational_query(chat_req.question):
        reply = get_conversational_response()
        await websocket.send_text(json.dumps({"type": "sources", "sources": []}))
        await websocket.send_text(json.dumps({"type": "token", "token": reply}))
        await websocket.send_text(json.dumps({"type": "done"}))
        await websocket.close()
        return

    async def send_status(phase: str) -> None:
        await websocket.send_text(json.dumps({"type": "status", "phase": phase}))

    await send_status("loading_index")
    t0 = time.perf_counter()

    try:
        retriever = await _build_retriever(chat_req.repo_id, embed_service, chroma_client)
        corpus_size = retriever._collection.count()  # noqa: SLF001

        async def retrieval_status(phase: str) -> None:
            await send_status(phase)

        chunks = await _retrieve_with_timeout(
            retriever,
            chat_req.question,
            settings.RETRIEVAL_TIMEOUT,
            status_callback=retrieval_status,
        )
    except asyncio.TimeoutError:
        logger.error("WS retrieval timed out for repo_id=%s", chat_req.repo_id)
        await websocket.send_text(
            json.dumps({
                "type": "error",
                "message": f"Retrieval timed out after {settings.RETRIEVAL_TIMEOUT}s",
            })
        )
        await websocket.close()
        return
    except Exception as exc:
        logger.exception(
            "WS retrieval failed for repo_id=%s: %s", chat_req.repo_id, exc
        )
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        await websocket.close()
        return

    retrieval_ms = (time.perf_counter() - t0) * 1000
    log_timing(
        "routes_query.ws_retrieve",
        retrieval_ms,
        {"repo_id": chat_req.repo_id, "chunk_count": len(chunks)},
    )

    sources_payload = [
        {
            "file_path": c.get("file_path", ""),
            "start_line": int(c.get("start_line", 0)),
            "end_line": int(c.get("end_line", 0)),
            "chunk_type": c.get("chunk_type", "snippet"),
            "symbol_name": c.get("symbol_name", ""),
            "text": c.get("text", ""),
        }
        for c in chunks
    ]
    await websocket.send_text(json.dumps({"type": "sources", "sources": sources_payload}))

    sufficient = has_sufficient_context(chunks, corpus_size=corpus_size)
    agent_debug_log(
        "routes_query.py:ws_chat",
        "Guardrail decision",
        {
            "repo_id": chat_req.repo_id,
            "question_prefix": chat_req.question[:80],
            "chunk_count": len(chunks),
            "corpus_size": corpus_size,
            "max_score": max((c.get("score", 0.0) for c in chunks), default=0.0),
            "sufficient": sufficient,
            "retrieval_ms": round(retrieval_ms, 2),
        },
        hypothesis_id="H1",
    )
    if not sufficient:
        await websocket.send_text(
            json.dumps({"type": "token", "token": _NO_CONTEXT_RESPONSE})
        )
        await websocket.send_text(json.dumps({"type": "done"}))
        await websocket.close()
        return

    messages = build_prompt(chat_req.question, chunks)

    await send_status("generating")

    try:
        async for token in ollama_client.stream_response(messages, model):
            await websocket.send_text(json.dumps({"type": "token", "token": token}))
    except WebSocketDisconnect:
        logger.info("WS: client disconnected during streaming")
        return
    except Exception as exc:
        logger.exception("WS: streaming error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await websocket.close()
        except Exception:
            pass
        return

    await websocket.send_text(json.dumps({"type": "done"}))
    await websocket.close()
    logger.info("WS chat: completed for repo_id=%s", chat_req.repo_id)
