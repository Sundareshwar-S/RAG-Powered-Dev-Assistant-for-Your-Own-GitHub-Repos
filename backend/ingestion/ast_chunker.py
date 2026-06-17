"""AST-based code chunker using tree-sitter 0.23.x.

Key API notes for tree-sitter 0.23.x:
- Grammar packages expose a ``language()`` (or ``language_typescript()``) capsule.
- ``Language(capsule)`` wraps the capsule into a ``Language`` object.
- ``Parser(language)`` accepts the ``Language`` directly — no ``set_language()`` needed.
- ``build_library()`` was removed in 0.22 and is NOT used here.
"""
from __future__ import annotations

from typing import Optional

import tiktoken
from tree_sitter import Language, Node, Parser

import tree_sitter_go as ts_go
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_javascript
import tree_sitter_python as ts_python
import tree_sitter_rust as ts_rust
import tree_sitter_typescript as ts_typescript

from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, Language] = {
    "python": Language(ts_python.language()),
    "javascript": Language(ts_javascript.language()),
    "typescript": Language(ts_typescript.language_typescript()),
    "java": Language(ts_java.language()),
    "go": Language(ts_go.language()),
    "rust": Language(ts_rust.language()),
}

# Languages that use the sliding-window fallback (no AST parser).
FALLBACK_LANGUAGES: frozenset[str] = frozenset({"markdown", "plaintext"})

ALL_SUPPORTED: frozenset[str] = frozenset(LANGUAGE_MAP) | FALLBACK_LANGUAGES

# ---------------------------------------------------------------------------
# Node type targets per language
# ---------------------------------------------------------------------------

TARGET_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {"function_definition", "class_definition", "decorated_definition"}
    ),
    "javascript": frozenset(
        {"function_declaration", "arrow_function", "class_declaration"}
    ),
    "typescript": frozenset(
        {"function_declaration", "arrow_function", "class_declaration"}
    ),
    "java": frozenset({"method_declaration", "class_declaration"}),
    "go": frozenset({"function_declaration", "method_declaration"}),
    "rust": frozenset({"function_item", "impl_item"}),
}

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

CHUNK_TOKEN_LIMIT: int = 512
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TIKTOKEN_ENC.encode(text))


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class ASTChunker:
    """Splits a source file into semantically meaningful chunks.

    For parseable languages (Python, JS, TS, Java, Go, Rust) the tree-sitter
    AST is used to extract top-level functions, classes, and methods.  For
    markdown and plaintext a sliding-window fallback is used.

    Each returned chunk dict contains:
        text, file_path, language, chunk_type,
        start_line (1-indexed), end_line (1-indexed), symbol_name
    """

    def __init__(self, language: str) -> None:
        if language not in ALL_SUPPORTED:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported: {sorted(ALL_SUPPORTED)}"
            )
        self.language = language
        self._target_types: frozenset[str] = TARGET_NODE_TYPES.get(
            language, frozenset()
        )
        self._parser: Optional[Parser] = (
            Parser(LANGUAGE_MAP[language])
            if language in LANGUAGE_MAP
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        if self.language in FALLBACK_LANGUAGES:
            return self._sliding_window_fallback(source_code, file_path)

        try:
            tree = self._parser.parse(source_code.encode("utf-8"))  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "AST parse failed for %s (%s), using sliding window: %s",
                file_path,
                self.language,
                exc,
            )
            return self._sliding_window_fallback(source_code, file_path)

        chunks: list[dict] = []
        self._traverse(tree.root_node, source_code, file_path, chunks)

        if not chunks:
            # Parsed successfully but no matching top-level nodes found
            # (e.g. a file that only has imports/assignments).
            chunks = self._sliding_window_fallback(source_code, file_path)

        return chunks

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def _traverse(
        self,
        node: Node,
        source: str,
        file_path: str,
        out: list[dict],
    ) -> None:
        if node.type in self._target_types:
            lines = source.split("\n")
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            text = "\n".join(lines[start_line : end_line + 1])

            if count_tokens(text) <= CHUNK_TOKEN_LIMIT:
                out.append(
                    {
                        "text": text,
                        "file_path": file_path,
                        "language": self.language,
                        "chunk_type": node.type,
                        "start_line": start_line + 1,  # convert to 1-indexed
                        "end_line": end_line + 1,
                        "symbol_name": self._extract_name(node, source),
                    }
                )
                # Do not recurse into matched nodes to avoid duplicate chunks
                # (e.g. a method inside a class is captured by its own pass).
                return

        for child in node.children:
            self._traverse(child, source, file_path, out)

    def _extract_name(self, node: Node, source: str) -> str:
        """Return the identifier/name of a node, or '<anonymous>'."""
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier"):
                return source[child.start_byte : child.end_byte]
        return "<anonymous>"

    # ------------------------------------------------------------------
    # Fallback: sliding window
    # ------------------------------------------------------------------

    def _sliding_window_fallback(
        self, source: str, file_path: str
    ) -> list[dict]:
        """Split non-parseable text into overlapping line windows."""
        lines = source.split("\n")
        window = 60
        stride = 45
        chunks: list[dict] = []

        i = 0
        while i < len(lines):
            window_lines = lines[i : i + window]
            text = "\n".join(window_lines)
            chunks.append(
                {
                    "text": text,
                    "file_path": file_path,
                    "language": self.language,
                    "chunk_type": "sliding_window",
                    "start_line": i + 1,
                    "end_line": min(i + window, len(lines)),
                    "symbol_name": "",
                }
            )
            i += stride

        return chunks
