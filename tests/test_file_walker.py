"""Unit tests for FileWalker extension and skip rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def test_file_walker_indexes_config_and_shell_files(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "config.yaml").write_text("key: value")
    (tmp_path / "Dockerfile").write_text("FROM python:3.11")
    (tmp_path / "run.sh").write_text("#!/bin/bash")

    results = FileWalker().walk(tmp_path)
    found = {rel for rel, _, _ in results}

    assert found == {"app.py", "config.yaml", "Dockerfile", "run.sh"}


def test_file_walker_skips_html_and_css(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "page.html").write_text("<html></html>")
    (tmp_path / "styles.css").write_text("body {}")
    (tmp_path / "ui" / "static").mkdir(parents=True)
    (tmp_path / "ui" / "static" / "app.js").write_text("console.log('hi')")

    results = FileWalker().walk(tmp_path)
    found = {rel for rel, _, _ in results}

    assert found == {"main.py"}


def test_file_walker_skips_excluded_dirs(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "lib.py").write_text("y = 2")

    for skip in (".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"):
        skip_dir = tmp_path / skip
        skip_dir.mkdir()
        (skip_dir / "evil.py").write_text("evil = True")

    results = FileWalker().walk(tmp_path)
    found = [rel for rel, _, _ in results]

    assert len(found) == 2
    assert all(
        not any(skip in p for skip in (".git", "node_modules", "__pycache__", "venv", "dist"))
        for p in found
    )
