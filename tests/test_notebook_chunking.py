"""Unit tests for Jupyter notebook chunking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def _sample_notebook() -> str:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# NASA anomaly detection\n", "Overview of the pipeline."],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')\n"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["def preprocess(x):\n", "    return x.dropna()\n"],
            },
        ],
    }
    return json.dumps(notebook)


class TestNotebookChunking:
    def test_notebook_extracts_code_and_markdown_cells(self) -> None:
        from ingestion.notebook_chunker import NotebookChunker

        chunks = NotebookChunker().chunk(_sample_notebook(), "analysis.ipynb")

        assert len(chunks) == 3
        assert chunks[0]["chunk_type"] == "notebook_cell"
        assert chunks[0]["language"] == "markdown"
        assert "NASA anomaly detection" in chunks[0]["text"]
        assert chunks[1]["language"] == "python"
        assert "pandas" in chunks[1]["text"]
        assert chunks[2]["symbol_name"] == "cell_3_code"
        assert "preprocess" in chunks[2]["text"]

    def test_invalid_notebook_returns_empty(self) -> None:
        from ingestion.notebook_chunker import NotebookChunker

        chunks = NotebookChunker().chunk("not json", "broken.ipynb")

        assert chunks == []

    def test_notebook_indexes_all_cells_when_cap_is_zero(self, monkeypatch) -> None:
        monkeypatch.setattr("core.config.settings.NOTEBOOK_MAX_CELLS", 0)

        from ingestion.notebook_chunker import NotebookChunker

        cells = []
        for i in range(60):
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [f"x_{i} = {i}\n"],
                }
            )
        notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells}
        import json

        chunks = NotebookChunker().chunk(json.dumps(notebook), "big.ipynb")

        assert len(chunks) == 60
