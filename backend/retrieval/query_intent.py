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

_FILE_EXT_RE = re.compile(
    r"([\w./\\-]+\."
    r"(?:md|txt|py|js|ts|jsx|tsx|java|go|rs|yaml|yml|json|toml|ini|cfg|"
    r"sh|bash|html|htm|jinja2?))"
    r"(?:\s+file)?",
    re.IGNORECASE,
)

_README_WORD_RE = re.compile(
    r"\breadme\b(?!\s+(?:folder|directory|dir)\b)",
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
        "readme",
        "the",
        "structure",
    }
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().strip("/")


def _mentions_specific_file(query: str) -> bool:
    """Return True when the query names a specific file rather than a folder."""
    if _FILE_EXT_RE.search(query):
        return True
    return bool(_README_WORD_RE.search(query))


def extract_target_file(query: str, known_paths: list[str]) -> str | None:
    """Match a filename reference in *query* against *known_paths*."""
    text = query.strip()
    if not text or not known_paths:
        return None

    candidates: list[str] = []
    for match in _FILE_EXT_RE.finditer(text):
        candidates.append(_normalize_path(match.group(1)))

    if _README_WORD_RE.search(text):
        candidates.append("readme")

    if not candidates:
        return None

    path_lookup = {_normalize_path(p).lower(): p for p in known_paths}
    basename_lookup: dict[str, str] = {}
    for path in known_paths:
        basename_lookup[path.split("/")[-1].lower()] = path

    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in path_lookup:
            return path_lookup[normalized]

        basename = candidate.split("/")[-1].lower()
        if basename in basename_lookup:
            return basename_lookup[basename]

        for path in known_paths:
            lower_path = _normalize_path(path).lower()
            if lower_path == normalized or lower_path.endswith("/" + normalized):
                return path

    return None


def is_file_content_query(query: str, known_paths: list[str] | None = None) -> bool:
    """Return True when the query asks about a specific file's content."""
    if known_paths:
        return extract_target_file(query, known_paths) is not None
    return _mentions_specific_file(query)


def is_structure_query(query: str) -> bool:
    """Return True for repo-structure / file-listing questions."""
    text = query.strip()
    if not text:
        return False
    if _mentions_specific_file(text):
        return False
    if _STRUCTURE_RE.search(text):
        return True
    # Typos / informal phrasing: "whare are inside the backend folder"
    if re.search(
        r"\b(?:inside|in)\s+(?:the\s+)?[\w./-]+\s+(?:folder|directory|dir)\b",
        text,
        re.I,
    ):
        return True
    if re.search(r"\b(?:files?|contents?)\s+(?:in|inside|under)\b", text, re.I):
        return True
    return False


_STRUCTURE_PHRASES: tuple[str, ...] = (
    "file structure",
    "repo structure",
    "repository structure",
    "directory structure",
    "project structure",
    "code structure",
    "folder structure",
)


def _strip_structure_phrases(text: str) -> str:
    """Remove compound structure phrases so folder regex does not misfire."""
    normalized = text
    for phrase in _STRUCTURE_PHRASES:
        normalized = re.sub(re.escape(phrase), " ", normalized, flags=re.IGNORECASE)
    return normalized


def extract_folder_prefix(query: str) -> str | None:
    """Extract a folder prefix from a structure query, if present."""
    text = query.strip()
    normalized = _strip_structure_phrases(text)

    # Repo-wide structure questions ("file structure", etc.) are not folder-scoped.
    if normalized != text and not re.search(
        r"\b(?:folder|directory|dir)\b|[\w./-]+/",
        text,
        re.I,
    ):
        return None

    match = _FOLDER_RE.search(normalized)
    if not match:
        return None
    candidate = match.group(1).strip().strip("/").replace("\\", "/")
    if not candidate or candidate.lower() in _NON_FOLDER_WORDS:
        return None
    if "." in candidate.split("/")[-1]:
        return None
    return candidate
