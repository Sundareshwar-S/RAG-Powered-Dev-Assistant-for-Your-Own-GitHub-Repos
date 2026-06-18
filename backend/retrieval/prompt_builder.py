"""Prompt construction for the CodeBase Oracle LLM generation step."""
from __future__ import annotations

import re

import tiktoken

from core.config import settings
from core.logger import get_logger
from retrieval.query_intent import is_structure_query

logger = get_logger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

_PROMPT_FORMATTING_OVERHEAD = 600
MIN_DENSE_SCORE_THRESHOLD = 0.35

MIN_SIMILARITY_THRESHOLD = 0.10

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
- When a repository file index section is provided, treat it as the complete \
list of searchable source files. Distinguish three sections when present: \
(1) indexed searchable files, (2) partially indexed/truncated files, \
(3) excluded binary/non-text assets listed for structure only.
- When asked about repo contents or structure, enumerate indexed searchable \
files from the file index. Mention truncated and excluded asset paths when \
those sections are listed.
- When code chunks are provided for a named file, answer from that chunk content. \
Do not claim a file is empty when its text appears in the provided context.
"""


def context_token_limit() -> int:
    """Chunk token budget derived from OLLAMA_NUM_CTX."""
    return settings.OLLAMA_NUM_CTX - 1200


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

    if any(c.get("chunk_type") == "file_manifest" for c in chunks):
        return True

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


def _format_structure_index(chunk: dict) -> str:
    """Format a file-index chunk for structure questions."""
    return (
        "Repository file index:\n"
        f"{chunk.get('text', '')}"
    )


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build an Ollama ``/api/chat`` messages list from *query* and *chunks*."""
    budget = context_token_limit()
    structure = is_structure_query(query)

    if structure:
        index_chunks = [
            c for c in chunks if c.get("chunk_type") == "file_manifest"
        ]
        if not index_chunks:
            index_chunks = chunks
        trimmed = _trim_structure_chunks(index_chunks, budget)
        chunk_blocks = "\n\n".join(
            _format_structure_index(chunk) for chunk in trimmed
        )
    else:
        formatter = _format_chunk
        trimmed = _trim_to_budget(chunks, budget)
        if len(trimmed) < len(chunks):
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


def _trim_structure_chunks(chunks: list[dict], budget: int) -> list[dict]:
    """Keep structure index chunks within the token budget."""
    total = 0
    kept: list[dict] = []
    for chunk in chunks:
        formatted = _format_structure_index(chunk)
        tokens = count_tokens(formatted)
        if total + tokens > budget:
            break
        kept.append(chunk)
        total += tokens
    return kept or chunks[:1]


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
