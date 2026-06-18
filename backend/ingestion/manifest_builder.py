"""Build searchable file-tree manifest chunks for indexed repositories."""
from __future__ import annotations

from collections import defaultdict

MANIFEST_PATH = "__manifest__/repo_tree.txt"
MAX_PATHS_PER_CHUNK = 400


def build_manifest_chunks(
    indexed_paths: list[str],
    *,
    skipped_summary: str = "",
    truncated_paths: list[str] | None = None,
) -> list[dict]:
    """Return manifest chunk dicts describing the indexed file tree.

    For large repos the tree is split into multiple ``file_manifest`` chunks.
    """
    paths = sorted({p.replace("\\", "/") for p in indexed_paths if p})
    if not paths and not skipped_summary:
        return []

    chunks: list[dict] = []
    batches = [paths[i : i + MAX_PATHS_PER_CHUNK] for i in range(0, len(paths), MAX_PATHS_PER_CHUNK)]
    if not batches:
        batches = [[]]

    for batch_index, batch in enumerate(batches):
        tree_text = format_tree(batch) if batch else "(no searchable files)"
        header = (
            f"Repository file index — searchable text files ({len(paths)} total):\n"
        )
        text = header + tree_text

        if batch_index == 0:
            if truncated_paths:
                text += (
                    "\n\nPartially indexed (first "
                    f"{len(truncated_paths)} file(s) truncated to size limit):\n"
                    + "\n".join(f"  {p}" for p in sorted(truncated_paths))
                )
            if skipped_summary:
                text += (
                    "\n\nExcluded from search (binary/non-text — on disk only):\n"
                    + skipped_summary
                )

        chunks.append(
            {
                "text": text,
                "file_path": MANIFEST_PATH,
                "language": "plaintext",
                "chunk_type": "file_manifest",
                "start_line": 1,
                "end_line": text.count("\n") + 1,
                "symbol_name": "",
            }
        )
    return chunks


def format_tree(paths: list[str]) -> str:
    """Format sorted relative paths as an indented directory tree."""
    tree: dict = {}
    for path in sorted(paths):
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None

    lines: list[str] = []

    def _walk(node: dict, prefix: str) -> None:
        dirs = sorted(k for k in node if k.endswith("/"))
        files = sorted(k for k in node if not k.endswith("/"))
        for directory in dirs:
            lines.append(f"{prefix}{directory}")
            _walk(node[directory], prefix + "  ")
        for filename in files:
            lines.append(f"{prefix}{filename}")

    _walk(tree, "")
    return "\n".join(lines)


def dedupe_indexed_paths(corpus: list[dict]) -> list[str]:
    """Collect unique indexed file paths from a chunk corpus."""
    paths: set[str] = set()
    for chunk in corpus:
        path = chunk.get("file_path", "")
        if not path or path.startswith("__manifest__/"):
            continue
        paths.add(path.replace("\\", "/"))
    return sorted(paths)


def filter_paths_by_prefix(paths: list[str], prefix: str) -> list[str]:
    """Return paths under *prefix* (case-insensitive, normalized)."""
    normalized = prefix.replace("\\", "/").strip().strip("/")
    if not normalized:
        return paths
    needle = normalized.lower() + "/"
    filtered = [
        p
        for p in paths
        if p.lower() == normalized.lower() or p.lower().startswith(needle)
    ]
    return filtered


def build_folder_tree_text(paths: list[str], folder_prefix: str = "") -> str:
    """Build a compact tree string for *paths*, optionally scoped to a folder."""
    if folder_prefix:
        scoped = filter_paths_by_prefix(paths, folder_prefix)
        if not scoped:
            return f"(No indexed files under {folder_prefix}/)"
        rel_paths: list[str] = []
        prefix_norm = folder_prefix.strip("/").replace("\\", "/")
        for path in scoped:
            lower_path = path.lower()
            lower_prefix = prefix_norm.lower()
            if lower_path == lower_prefix:
                continue
            if lower_path.startswith(lower_prefix + "/"):
                rel_paths.append(path[len(prefix_norm) + 1 :])
            else:
                rel_paths.append(path)
        root = prefix_norm + "/"
        body = format_tree(rel_paths) if rel_paths else "(no files)"
        return f"{root}\n{body}"
    return format_tree(paths)


def group_paths_by_top_level(paths: list[str]) -> dict[str, list[str]]:
    """Group paths by their first path component (for diagnostics)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        top = path.split("/", 1)[0]
        groups[top].append(path)
    return dict(groups)
