"""Jupyter notebook chunker — extracts code and markdown cells from .ipynb files."""
from __future__ import annotations

import json
from typing import Any

from core.config import settings
from ingestion.ast_chunker import (
    CHUNK_TOKEN_LIMIT,
    count_tokens,
    sliding_window_cap_for_file,
)

NOTEBOOK_LANGUAGES = frozenset({"notebook"})


class NotebookChunker:
    """Split a Jupyter notebook into searchable cell chunks."""

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        try:
            notebook = json.loads(source_code)
        except json.JSONDecodeError:
            return []

        cells: list[dict[str, Any]] = notebook.get("cells", [])
        if not isinstance(cells, list):
            return []

        max_cells = settings.NOTEBOOK_MAX_CELLS
        chunks: list[dict] = []

        for index, cell in enumerate(cells[:max_cells]):
            if not isinstance(cell, dict):
                continue

            cell_type = cell.get("cell_type", "")
            if cell_type not in ("code", "markdown"):
                continue

            source = cell.get("source", "")
            if isinstance(source, list):
                text = "".join(source)
            elif isinstance(source, str):
                text = source
            else:
                continue

            text = text.strip()
            if not text:
                continue

            language = "python" if cell_type == "code" else "markdown"
            symbol_name = f"cell_{index + 1}_{cell_type}"
            chunk_type = "notebook_cell"

            if count_tokens(text) <= CHUNK_TOKEN_LIMIT:
                line_count = text.count("\n") + 1
                chunks.append(
                    {
                        "text": text,
                        "file_path": file_path,
                        "language": language,
                        "chunk_type": chunk_type,
                        "start_line": index + 1,
                        "end_line": index + line_count,
                        "symbol_name": symbol_name,
                    }
                )
                continue

            chunks.extend(
                self._split_long_cell(
                    text,
                    file_path,
                    language,
                    chunk_type,
                    symbol_name,
                    index + 1,
                )
            )

        return chunks

    def _split_long_cell(
        self,
        text: str,
        file_path: str,
        language: str,
        chunk_type: str,
        symbol_name: str,
        cell_index: int,
    ) -> list[dict]:
        lines = text.split("\n")
        window = 60
        stride = 45
        max_chunks = sliding_window_cap_for_file(file_path)
        chunks: list[dict] = []
        i = 0

        while i < len(lines):
            if max_chunks > 0 and len(chunks) >= max_chunks:
                break
            window_end = min(i + window - 1, len(lines) - 1)
            window_text = "\n".join(lines[i : window_end + 1]).strip()
            if window_text:
                chunks.append(
                    {
                        "text": window_text,
                        "file_path": file_path,
                        "language": language,
                        "chunk_type": chunk_type,
                        "start_line": cell_index,
                        "end_line": cell_index + window_end - i,
                        "symbol_name": symbol_name,
                    }
                )
            i += stride
            if i >= len(lines):
                break

        return chunks
