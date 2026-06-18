"""AST-based code chunker using tree-sitter 0.23.x.

Key API notes for tree-sitter 0.23.x:
- Grammar packages expose a ``language()`` (or ``language_typescript()``) capsule.
- ``Language(capsule)`` wraps the capsule into a ``Language`` object.
- ``Parser(language)`` accepts the ``Language`` directly — no ``set_language()`` needed.
- ``build_library()`` was removed in 0.22 and is NOT used here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import tiktoken
from tree_sitter import Language, Node, Parser

import tree_sitter_go as ts_go
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_javascript
import tree_sitter_python as ts_python
import tree_sitter_rust as ts_rust
import tree_sitter_typescript as ts_typescript

from core.config import settings
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

DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".sh",
        ".bash",
        ".csv",
        ".tsv",
        ".sql",
        ".r",
        ".xml",
        ".env",
        ".log",
        ".rst",
        ".tex",
        ".bib",
        ".gradle",
        ".properties",
        ".css",
        ".scss",
        ".sass",
        ".less",
    }
)
HTML_EXTENSIONS: frozenset[str] = frozenset({".html", ".htm", ".jinja", ".jinja2"})
CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}
)

# Container nodes: recurse into children for method-level chunks when small.
CLASS_NODE_TYPES: frozenset[str] = frozenset(
    {"class_definition", "class_declaration", "impl_item"}
)


def sliding_window_cap_for_file(file_path: str) -> int:
    """Return max sliding-window chunks for *file_path* (0 means unlimited)."""
    suffix = Path(file_path).suffix.lower()
    basename = Path(file_path).name.lower()
    if suffix in HTML_EXTENSIONS:
        return settings.HTML_MAX_SLIDING_CHUNKS
    if suffix in DOC_EXTENSIONS or basename == "readme":
        return settings.DOC_MAX_SLIDING_CHUNKS
    if suffix in CODE_EXTENSIONS:
        return settings.AST_MAX_SLIDING_CHUNKS
    return settings.HTML_MAX_SLIDING_CHUNKS


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
        if self.language == "python" and node.type == "if_statement":
            lines = source.split("\n")
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            text = "\n".join(lines[start_line : end_line + 1])
            if "__name__" in text and "__main__" in text:
                if count_tokens(text) <= CHUNK_TOKEN_LIMIT:
                    out.append(
                        {
                            "text": text,
                            "file_path": file_path,
                            "language": self.language,
                            "chunk_type": "main_guard",
                            "start_line": start_line + 1,
                            "end_line": end_line + 1,
                            "symbol_name": "__main__",
                        }
                    )
                else:
                    out.extend(
                        self._sliding_window_on_range(
                            lines,
                            start_line,
                            end_line,
                            file_path,
                            "main_guard",
                            "__main__",
                        )
                    )
                return

        if node.type in self._target_types:
            lines = source.split("\n")
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            text = "\n".join(lines[start_line : end_line + 1])

            if node.type in CLASS_NODE_TYPES:
                if count_tokens(text) > CHUNK_TOKEN_LIMIT:
                    out.extend(
                        self._sliding_window_on_range(
                            lines,
                            start_line,
                            end_line,
                            file_path,
                            node.type,
                            self._extract_name(node, source),
                        )
                    )
                else:
                    for child in node.children:
                        self._traverse(child, source, file_path, out)
                return

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
                return

            out.extend(
                self._sliding_window_on_range(
                    lines,
                    start_line,
                    end_line,
                    file_path,
                    node.type,
                    self._extract_name(node, source),
                )
            )
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
        return self._sliding_window_on_range(
            lines, 0, len(lines) - 1, file_path, "sliding_window", ""
        )

    def _sliding_window_on_range(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
        file_path: str,
        chunk_type: str,
        symbol_name: str,
    ) -> list[dict]:
        """Split a line range into overlapping windows."""
        if end_line < start_line:
            return []

        window = 60
        stride = 45
        max_chunks = sliding_window_cap_for_file(file_path)
        chunks: list[dict] = []
        i = start_line

        while i <= end_line:
            if max_chunks > 0 and len(chunks) >= max_chunks:
                break
            if settings.MAX_CHUNKS_PER_FILE > 0 and len(chunks) >= settings.MAX_CHUNKS_PER_FILE:
                break
            window_end = min(i + window - 1, end_line)
            text = "\n".join(lines[i : window_end + 1])
            if text.strip():
                chunks.append(
                    {
                        "text": text,
                        "file_path": file_path,
                        "language": self.language,
                        "chunk_type": chunk_type,
                        "start_line": i + 1,
                        "end_line": window_end + 1,
                        "symbol_name": symbol_name,
                    }
                )
            i += stride
            if i > end_line:
                break

        return chunks
