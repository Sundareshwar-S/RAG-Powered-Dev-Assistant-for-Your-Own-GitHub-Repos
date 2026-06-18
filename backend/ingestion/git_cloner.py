import hashlib
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
from git import Repo
from git.exc import GitCommandError

from core.config import settings
from core.debug_log import agent_debug_log
from core.logger import get_logger

logger = get_logger(__name__)

REPOS_BASE = Path("/tmp/repos")

# Regex to extract owner/repo from a GitHub HTTPS URL.
_GITHUB_REPO_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


class GitCloner:
    """Shallow-clones a GitHub repo into /tmp/repos/{repo_id}.

    Re-uses an existing clone by pulling the latest commit if the
    directory is already present.  Injects GITHUB_TOKEN into the URL
    for private HTTPS repos when the env var is set.

    Repo size guard
    ---------------
    Before any network clone is attempted, ``clone()`` calls the GitHub REST
    API to check the reported repository size.  Repos larger than
    ``settings.MAX_REPO_SIZE_MB`` are rejected with a ``ValueError`` so the
    ingestion job fails fast rather than exhausting disk space.
    """

    def clone(self, repo_url: str, branch: str = "main") -> tuple[Path, str]:
        repo_url = self._normalize_repo_url(repo_url)
        self._check_repo_size(repo_url)

        repo_id = self._make_repo_id(repo_url)
        clone_path = REPOS_BASE / repo_id
        auth_url = self._inject_token(repo_url)

        if clone_path.exists() and (clone_path / ".git").exists():
            logger.info("Repo %s already cloned — checking out %s and pulling", repo_id, branch)
            repo = Repo(clone_path)
            try:
                repo.git.checkout(branch)
            except GitCommandError as exc:
                logger.warning(
                    "Could not checkout branch=%s for %s (%s); continuing on current branch",
                    branch,
                    repo_id,
                    exc,
                )
            repo.remotes.origin.pull()
        else:
            clone_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Cloning %s (branch=%s) → %s", repo_url, branch, clone_path
            )
            try:
                Repo.clone_from(auth_url, clone_path, branch=branch, depth=1)
            except GitCommandError as exc:
                logger.warning(
                    "Clone with branch=%s failed for %s (%s); retrying default branch",
                    branch,
                    repo_url,
                    exc,
                )
                agent_debug_log(
                    "git_cloner.py:clone",
                    "Branch clone failed; retrying default branch",
                    {"repo_url": repo_url, "branch": branch, "error": str(exc)},
                    hypothesis_id="H5",
                )
                shutil.rmtree(clone_path, ignore_errors=True)
                clone_path.mkdir(parents=True, exist_ok=True)
                Repo.clone_from(auth_url, clone_path, depth=1)

        agent_debug_log(
            "git_cloner.py:clone",
            "Clone succeeded",
            {"repo_url": repo_url, "branch": branch, "repo_id": repo_id},
            hypothesis_id="H5",
        )
        return clone_path, repo_id

    # ------------------------------------------------------------------
    # Repo size guard
    # ------------------------------------------------------------------

    def _check_repo_size(self, repo_url: str) -> None:
        """Raise ValueError if the repo exceeds MAX_REPO_SIZE_MB.

        Uses the GitHub REST API for github.com URLs; silently skips the
        check for other hosts or when the API call fails (to avoid blocking
        on network errors or rate limits).
        """
        size_mb = self._fetch_repo_size_mb(repo_url)
        if size_mb is None:
            return  # could not determine size — allow the clone to proceed

        limit_mb = settings.MAX_REPO_SIZE_MB
        if size_mb > limit_mb:
            raise ValueError(
                f"Repository is too large: {size_mb} MB "
                f"(limit is {limit_mb} MB). "
                "Reduce MAX_REPO_SIZE_MB or choose a smaller repo."
            )
        logger.info("Repo size check passed: %s MB (limit %s MB)", size_mb, limit_mb)

    def _fetch_repo_size_mb(self, repo_url: str) -> int | None:
        """Return the repository size in MB via the GitHub API, or None.

        GitHub reports size in kilobytes in the ``size`` field of the
        repository metadata endpoint.

        Returns None if the URL is not a github.com URL, if the token is
        missing for a private repo, or if any network/HTTP error occurs.
        """
        match = _GITHUB_REPO_RE.match(repo_url)
        if not match:
            return None  # non-GitHub host — skip size check

        owner = match.group("owner")
        repo = match.group("repo")
        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(api_url, headers=headers)
                response.raise_for_status()
                size_kb: int = response.json().get("size", 0)
                # GitHub returns size in KB; convert to MB (round up).
                return max(1, (size_kb + 1023) // 1024)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "GitHub API returned %s for %s — skipping size check",
                exc.response.status_code,
                api_url,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch repo size for %s (%s) — skipping size check",
                repo_url,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_repo_url(repo_url: str) -> str:
        """Strip whitespace, trailing slashes, and a trailing ``.git`` suffix."""
        url = repo_url.strip().rstrip("/")
        if url.lower().endswith(".git"):
            url = url[:-4]
        return url

    @staticmethod
    def _make_repo_id(repo_url: str) -> str:
        return hashlib.md5(repo_url.encode()).hexdigest()[:8]

    @staticmethod
    def _inject_token(repo_url: str) -> str:
        if not settings.GITHUB_TOKEN:
            return repo_url
        parsed = urlparse(repo_url)
        if parsed.scheme in ("http", "https") and not parsed.username:
            netloc = f"{settings.GITHUB_TOKEN}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=netloc)
        return urlunparse(parsed)
