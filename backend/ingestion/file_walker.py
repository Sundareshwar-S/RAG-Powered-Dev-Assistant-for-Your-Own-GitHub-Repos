from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".txt": "plaintext",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".yaml": "plaintext",
    ".yml": "plaintext",
    ".json": "plaintext",
    ".toml": "plaintext",
    ".ini": "plaintext",
    ".cfg": "plaintext",
    ".sh": "plaintext",
    ".bash": "plaintext",
}

# Extensionless files indexed as plaintext when the basename matches exactly.
SUPPORTED_FILENAMES: dict[str, str] = {
    "dockerfile": "plaintext",
    "makefile": "plaintext",
    "jenkinsfile": "plaintext",
}

SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {".html", ".css", ".scss", ".sass", ".less"}
)

# Skip files under these relative path segments (low-value static assets).
SKIP_PATH_SEGMENTS: frozenset[str] = frozenset({"static", "assets"})

SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        "venv",
        ".venv",
        "dist",
        "build",
        "target",
        "coverage",
        ".tox",
        "vendor",
        ".next",
        "out",
    }
)


class FileWalker:
    """Recursively walks a repository directory and returns files whose
    extensions are in :data:`SUPPORTED_EXTENSIONS` or whose basename is in
    :data:`SUPPORTED_FILENAMES`.

    Directories listed in :data:`SKIP_DIRS` are pruned entirely so that
    vendor code, caches, and virtual-environments are never visited.
    HTML/CSS assets and files under ``static/`` / ``assets/`` are skipped.
    """

    def walk(self, repo_path: Path) -> list[tuple[str, str, str]]:
        """Return ``[(relative_path, full_path, language), ...]``."""
        results: list[tuple[str, str, str]] = []

        for full_path in repo_path.rglob("*"):
            if full_path.is_dir():
                continue
            if self._should_skip(full_path, repo_path):
                continue

            language = self._resolve_language(full_path)
            if language is None:
                continue

            relative = str(full_path.relative_to(repo_path))
            results.append((relative, str(full_path), language))

        logger.info("Found %d supported files in %s", len(results), repo_path)
        return results

    @staticmethod
    def _resolve_language(full_path: Path) -> str | None:
        suffix = full_path.suffix.lower()
        if suffix in SKIP_EXTENSIONS:
            return None
        if suffix in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[suffix]
        return SUPPORTED_FILENAMES.get(full_path.name.lower())

    @staticmethod
    def _should_skip(path: Path, base: Path) -> bool:
        """Return True if any component of the path (relative to base) is a
        directory that must be skipped."""
        try:
            relative_parts = path.relative_to(base).parts
        except ValueError:
            return False
        if frozenset(relative_parts) & SKIP_DIRS:
            return True
        return bool(frozenset(relative_parts) & SKIP_PATH_SEGMENTS)
