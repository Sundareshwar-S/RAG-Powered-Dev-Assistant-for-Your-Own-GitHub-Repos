"""CodeBase Oracle — FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes_ingest import router as ingest_router
from api.routes_query import router as query_router
from api.routes_repos import router as repos_router
from core.dependencies import get_embed_service
from core.config import settings
from core.limiter import limiter
from core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up in-process embed models when configured."""
    if settings.EMBED_BACKEND != "ollama":
        try:
            embed = get_embed_service()
            await embed.embed_batch(["warmup"], task="document")
            logger.info("Embedding model warmed up (%s)", settings.EMBED_BACKEND)
        except Exception as exc:
            logger.warning("Embed warmup skipped: %s", exc)
    else:
        logger.info("Using Ollama for embeddings — skipping in-process warmup")

    logger.info("CodeBase Oracle API ready")
    yield


app = FastAPI(
    title="CodeBase Oracle API",
    version="1.0.0",
    description="RAG-powered developer assistant for GitHub repositories",
    lifespan=lifespan,
)

# Attach the shared limiter so slowapi route decorators can find it
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server and any localhost origin used in Docker
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

# slowapi sends 429 with a structured body on rate limit exceeded
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 with a structured error envelope; never expose tracebacks."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "detail": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log the full traceback server-side, return a safe 500."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An internal server error occurred.",
            }
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(repos_router, prefix="/api/v1")

logger.info("CodeBase Oracle API starting up")
