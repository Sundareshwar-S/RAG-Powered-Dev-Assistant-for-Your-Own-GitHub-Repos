"""Shared query-intent detection for retrieval and prompt building."""
from __future__ import annotations

import re

_STRUCTURE_RE = re.compile(
    r"\b("
    r"what(?:'s|'s| is)?\s+(?:in|there in|inside)\s+(?:the\s+)?"
    r"(?:repo(?:sitory)?|[\w./-]+(?:\s+(?:folder|directory|dir))?)"
    r"|what\s+files?"
    r"|file\s+structure"
    r"|list\s+files?"
    r"|(?:show|tell me)\s+(?:the\s+)?(?:files?|contents?)"
    r"|contents?\s+of"
    r"|(?:inside|in)\s+(?:the\s+)?[\w./-]+\s+(?:folder|directory|dir)"
    r"|(?:overview|structure|layout|main\s+modules?|components)"
    r")\b",
    re.IGNORECASE,
)

_FOLDER_RE = re.compile(
    r"(?:"
    r"(?:what(?:'s|'s| is)?\s+(?:in|there in|inside)\s+(?:the\s+)?)"
    r"|(?:inside|in)\s+(?:the\s+)?"
    r"|(?:files?\s+(?:in|inside|under)\s+(?:the\s+)?)"
    r"|(?:contents?\s+of\s+(?:the\s+)?)"
    r")"
    r"([\w./-]+?)"
    r"(?:\s+(?:folder|directory|dir))?"
    r"(?:\s|$|\?)",
    re.IGNORECASE,
)

_NON_FOLDER_WORDS: frozenset[str] = frozenset(
    {
        "repo",
        "repository",
        "codebase",
        "project",
        "this",
        "that",
        "these",
        "those",
        "whole",
        "entire",
        "all",
    }
)


def is_structure_query(query: str) -> bool:
    """Return True for repo-structure / file-listing questions."""
    text = query.strip()
    if not text:
        return False
    if _STRUCTURE_RE.search(text):
        return True
    # Typos / informal phrasing: "whare are inside the backend folder"
    if re.search(r"\b(?:inside|in)\s+(?:the\s+)?[\w./-]+\s+(?:folder|directory|dir)\b", text, re.I):
        return True
    if re.search(r"\b(?:files?|contents?)\s+(?:in|inside|under)\b", text, re.I):
        return True
    return False


def extract_folder_prefix(query: str) -> str | None:
    """Extract a folder prefix from a structure query, if present."""
    text = query.strip()
    match = _FOLDER_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).strip().strip("/").replace("\\", "/")
    if not candidate or candidate.lower() in _NON_FOLDER_WORDS:
        return None
    return candidate
