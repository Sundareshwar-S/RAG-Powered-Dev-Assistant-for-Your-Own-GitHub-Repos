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
}

SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "node_modules", "__pycache__", "venv", ".venv"}
)


class FileWalker:
    """Recursively walks a repository directory and returns files whose
    extensions are in :data:`SUPPORTED_EXTENSIONS`.

    Directories listed in :data:`SKIP_DIRS` are pruned entirely so that
    vendor code, caches, and virtual-environments are never visited.
    """

    def walk(self, repo_path: Path) -> list[tuple[str, str, str]]:
        """Return ``[(relative_path, full_path, language), ...]``."""
        results: list[tuple[str, str, str]] = []

        for full_path in repo_path.rglob("*"):
            if full_path.is_dir():
                continue
            if self._should_skip(full_path, repo_path):
                continue

            language = SUPPORTED_EXTENSIONS.get(full_path.suffix.lower())
            if language is None:
                continue

            relative = str(full_path.relative_to(repo_path))
            results.append((relative, str(full_path), language))

        logger.info("Found %d supported files in %s", len(results), repo_path)
        return results

    @staticmethod
    def _should_skip(path: Path, base: Path) -> bool:
        """Return True if any component of the path (relative to base) is a
        directory that must be skipped."""
        try:
            relative_parts = path.relative_to(base).parts
        except ValueError:
            return False
        return bool(frozenset(relative_parts) & SKIP_DIRS)
