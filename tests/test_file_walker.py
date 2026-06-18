"""Unit tests for FileWalker extension and skip rules."""
from __future__ import annotations

import sys
from pathlib import Path

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


def test_file_walker_indexes_extensionless_readme(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "README").write_text("# Project")
    (tmp_path / "app.py").write_text("x = 1")

    results = FileWalker().walk(tmp_path)
    found = {rel for rel, _, lang in results}

    assert found == {"README", "app.py"}
    readme_langs = {lang for rel, _, lang in results if rel == "README"}
    assert readme_langs == {"markdown"}


def test_file_walker_indexes_html_and_static_js(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "index.html").write_text("<html><body>Hi</body></html>")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "app.js").write_text("console.log('hi')")
    (tmp_path / "styles.css").write_text("body {}")

    indexed, skipped = FileWalker().walk_with_stats(tmp_path)
    found = {rel for rel, _, _ in indexed}

    assert found == {"main.py", "templates/index.html", "static/app.js", "styles.css"}
    assert skipped == []


def test_file_walker_indexes_csv_and_unknown_text(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "data.csv").write_text("col1,col2\n1,2\n")
    (tmp_path / "custom.foo").write_text("custom text content\n")

    indexed, skipped = FileWalker().walk_with_stats(tmp_path)
    found = {rel for rel, _, _ in indexed}

    assert "data.csv" in found
    assert "custom.foo" in found
    assert skipped == []


def test_file_walker_skips_binary_assets_but_reports_them(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "static").mkdir(parents=True)
    (tmp_path / "static" / "plot.png").write_bytes(b"\x89PNG")
    (tmp_path / "model_library").mkdir()
    (tmp_path / "model_library" / "model.joblib").write_bytes(b"binary")

    indexed, skipped = FileWalker().walk_with_stats(tmp_path)
    found = {rel for rel, _, _ in indexed}

    assert found == {"main.py"}
    assert set(skipped) == {"static/plot.png", "model_library/model.joblib"}


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


def test_file_walker_summarize_skipped_groups_by_top_level() -> None:
    from ingestion.file_walker import FileWalker

    summary = FileWalker.summarize_skipped(
        [
            "static/plots/a.png",
            "static/plots/b.png",
            "model_library/model.keras",
        ]
    )

    assert "static/" in summary
    assert "model_library/" in summary
    assert "2 .png" in summary


def test_file_walker_indexes_notebooks(tmp_path: Path) -> None:
    from ingestion.file_walker import FileWalker

    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "analysis.ipynb").write_text('{"cells": [], "nbformat": 4}')

    results = FileWalker().walk(tmp_path)
    found = {rel: lang for rel, _, lang in results}

    assert found["main.py"] == "python"
    assert found["analysis.ipynb"] == "notebook"
