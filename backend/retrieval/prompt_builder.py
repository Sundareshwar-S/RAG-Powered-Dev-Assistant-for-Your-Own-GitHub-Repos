"""Prompt construction for the CodeBase Oracle LLM generation step."""
from __future__ import annotations

import re

import tiktoken

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

_PROMPT_FORMATTING_OVERHEAD = 600
MIN_DENSE_SCORE_THRESHOLD = 0.35

MIN_SIMILARITY_THRESHOLD = 0.10

_OVERVIEW_RE = re.compile(
    r"\b(what(?:'s|'s| is) (?:in|there in) (?:the )?repo|overview|structure|"
    r"what files|main modules|components|layout)\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_RE = re.compile(
    r"^(hi|hello|hey|howdy|thanks|thank you|bye|goodbye|good morning|good afternoon"
    r"|good evening|what(?:'s| is) your name|who are you)[!.?\s]*$",
    re.IGNORECASE,
)

_CONVERSATIONAL_RESPONSE = (
    "Hello! I'm CodeBase Oracle. I answer questions about your indexed codebase — "
    "functions, architecture, file structure, and how things work. "
    "What would you like to know?"
)

_SYSTEM_PROMPT = """\
You are CodeBase Oracle, an AI assistant that answers questions about software \
codebases by reasoning over indexed source code.

Rules:
- Always cite the exact file path and line range when referencing code.
- If the provided context does not contain enough information to answer the \
question, say so explicitly — do not guess or fabricate code.
- Prefer quoting short, relevant snippets over paraphrasing.
- Format code examples in fenced markdown code blocks with the language tag.
- If multiple files are relevant, address each one separately.
- When asked about repo contents or structure, enumerate all indexed files and \
modules from the provided chunks before diving into implementation details.\
"""


def context_token_limit() -> int:
    """Chunk token budget derived from OLLAMA_NUM_CTX."""
    return settings.OLLAMA_NUM_CTX - 1200


def is_overview_query(query: str) -> bool:
    return bool(_OVERVIEW_RE.search(query.strip()))


def is_conversational_query(query: str) -> bool:
    """Return True for greetings and other non-codebase questions."""
    return bool(_CONVERSATIONAL_RE.match(query.strip()))


def get_conversational_response() -> str:
    """Canned reply for conversational queries — no LLM call needed."""
    return _CONVERSATIONAL_RESPONSE


def count_tokens(text: str) -> int:
    """Return the approximate token count of *text* using cl100k_base."""
    return len(_enc.encode(text))


def has_sufficient_context(
    chunks: list[dict],
    corpus_size: int | None = None,
) -> bool:
    """Return True when retrieved chunks are strong enough to call the LLM."""
    if corpus_size is not None and corpus_size <= settings.SMALL_CORPUS_THRESHOLD:
        return bool(chunks)

    if not chunks:
        return False

    max_rerank = max(c.get("score", 0.0) for c in chunks)
    if max_rerank >= MIN_SIMILARITY_THRESHOLD:
        return True

    max_dense = max(c.get("dense_score", 0.0) for c in chunks)
    return max_dense >= MIN_DENSE_SCORE_THRESHOLD


def _format_chunk(chunk: dict, index: int) -> str:
    """Format a single chunk into a labelled code block for the prompt."""
    file_path = chunk.get("file_path", "unknown")
    start_line = chunk.get("start_line", "?")
    end_line = chunk.get("end_line", "?")
    chunk_type = chunk.get("chunk_type", "snippet")
    language = chunk.get("language", "")
    text = chunk.get("text", "")

    header = (
        f"[Chunk {index}] File: {file_path} "
        f"Lines {start_line}-{end_line} ({chunk_type})"
    )
    return f"{header}\n```{language}\n{text}\n```"


def _format_chunk_compact(chunk: dict, index: int) -> str:
    """Compact chunk summary for overview questions (paths + symbols, not full code)."""
    file_path = chunk.get("file_path", "unknown")
    start_line = chunk.get("start_line", "?")
    end_line = chunk.get("end_line", "?")
    chunk_type = chunk.get("chunk_type", "snippet")
    symbol_name = chunk.get("symbol_name", "")
    text = chunk.get("text", "")
    preview = text[:240].replace("\n", " ")
    if len(text) > 240:
        preview += "…"
    symbol_part = f" symbol={symbol_name}" if symbol_name else ""
    return (
        f"[Chunk {index}] File: {file_path} Lines {start_line}-{end_line} "
        f"({chunk_type}{symbol_part})\nPreview: {preview}"
    )


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build an Ollama ``/api/chat`` messages list from *query* and *chunks*."""
    budget = context_token_limit()
    overview = is_overview_query(query)

    if overview:
        trimmed = chunks
        formatter = _format_chunk_compact
    else:
        formatter = _format_chunk
        trimmed = _trim_to_budget(chunks, budget)

    if not overview and len(trimmed) < len(chunks):
        logger.info(
            "Context guard: trimmed %d → %d chunks to stay under %d tokens",
            len(chunks),
            len(trimmed),
            budget,
        )

    chunk_blocks = "\n\n".join(
        formatter(chunk, idx) for idx, chunk in enumerate(trimmed, start=1)
    )

    user_content = (
        f"Using the following code context, please answer the question.\n\n"
        f"{chunk_blocks}\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _trim_to_budget(chunks: list[dict], budget: int) -> list[dict]:
    """Include chunks while formatted representations fit in the token budget."""
    total = 0
    kept: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted = _format_chunk(chunk, idx)
        tokens = count_tokens(formatted)
        if total + tokens > budget:
            continue
        kept.append(chunk)
        total += tokens
    return kept
