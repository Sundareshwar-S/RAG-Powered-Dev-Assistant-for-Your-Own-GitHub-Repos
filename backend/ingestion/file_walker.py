from collections import Counter
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
    ".html": "plaintext",
    ".htm": "plaintext",
    ".jinja": "plaintext",
    ".jinja2": "plaintext",
    ".ipynb": "notebook",
}

# Extensionless files indexed as plaintext when the basename matches exactly.
SUPPORTED_FILENAMES: dict[str, str] = {
    "dockerfile": "plaintext",
    "makefile": "plaintext",
    "jenkinsfile": "plaintext",
    "readme": "markdown",
}

# Stylesheets are low-value for code search and are not reported as skipped.
SKIP_EXTENSIONS: frozenset[str] = frozenset({".css", ".scss", ".sass", ".less"})

# Binary / media / model artifacts — skipped regardless of directory.
SKIP_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".bmp",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp4",
        ".mp3",
        ".wav",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".pkl",
        ".pickle",
        ".joblib",
        ".h5",
        ".hdf5",
        ".keras",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyc",
        ".pyo",
    }
)

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
    """Recursively walks a repository directory and returns indexable files.

    Directories listed in :data:`SKIP_DIRS` are pruned entirely. Binary assets
    are skipped by extension. HTML/Jinja templates and text under ``static/``
    are indexed when they use supported extensions.
    """

    def walk(self, repo_path: Path) -> list[tuple[str, str, str]]:
        """Return ``[(relative_path, full_path, language), ...]``."""
        indexed, _ = self.walk_with_stats(repo_path)
        return indexed

    def walk_with_stats(
        self, repo_path: Path
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        """Return indexed files and relative paths skipped as binary/unsupported."""
        indexed: list[tuple[str, str, str]] = []
        skipped: list[str] = []

        for full_path in repo_path.rglob("*"):
            if full_path.is_dir():
                continue
            if self._is_under_skip_dir(full_path, repo_path):
                continue

            relative = str(full_path.relative_to(repo_path))
            suffix = full_path.suffix.lower()

            if suffix in SKIP_BINARY_EXTENSIONS:
                skipped.append(relative)
                continue

            language = self._resolve_language(full_path)
            if language is None:
                if suffix not in SKIP_EXTENSIONS:
                    skipped.append(relative)
                continue

            indexed.append((relative, str(full_path), language))

        logger.info(
            "Found %d supported files (%d skipped assets) in %s",
            len(indexed),
            len(skipped),
            repo_path,
        )
        return indexed, skipped

    @staticmethod
    def summarize_skipped(skipped_paths: list[str]) -> str:
        """Return a compact human-readable summary of skipped paths."""
        if not skipped_paths:
            return ""

        by_top: dict[str, list[str]] = {}
        for path in skipped_paths:
            top = path.split("/", 1)[0]
            by_top.setdefault(top, []).append(path)

        lines: list[str] = []
        for folder in sorted(by_top):
            paths = by_top[folder]
            ext_counts = Counter(Path(p).suffix.lower() or "(no ext)" for p in paths)
            ext_parts = ", ".join(
                f"{count} {ext}" for ext, count in ext_counts.most_common(4)
            )
            lines.append(f"  {folder}/ — {len(paths)} file(s) ({ext_parts})")
        return "\n".join(lines)

    @staticmethod
    def _resolve_language(full_path: Path) -> str | None:
        suffix = full_path.suffix.lower()
        if suffix in SKIP_EXTENSIONS:
            return None
        if suffix in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[suffix]
        return SUPPORTED_FILENAMES.get(full_path.name.lower())

    @staticmethod
    def _is_under_skip_dir(path: Path, base: Path) -> bool:
        try:
            relative_parts = path.relative_to(base).parts
        except ValueError:
            return False
        return bool(frozenset(relative_parts) & SKIP_DIRS)
