from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

# Repositories on these hosts are allowed; all others are rejected.
_ALLOWED_REPO_HOSTS: frozenset[str] = frozenset(
    {"github.com", "gitlab.com", "bitbucket.org"}
)


class IngestRequest(BaseModel):
    repo_url: HttpUrl
    branch: str = "main"

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: HttpUrl) -> HttpUrl:
        url_str = str(v)
        if url_str.startswith("http://"):
            raise ValueError(
                "Only HTTPS repository URLs are allowed. "
                f"Received: {url_str!r}"
            )
        host = urlparse(url_str).netloc.split(":")[0].lower()
        if host not in _ALLOWED_REPO_HOSTS:
            raise ValueError(
                f"Repository host {host!r} is not in the allowed list: "
                f"{sorted(_ALLOWED_REPO_HOSTS)}"
            )
        return v


class QueryRequest(BaseModel):
    repo_id: str
    question: str
    model: Optional[str] = "qwen2.5-coder:7b"


class SourceChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str
    text: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    sources: list[SourceChunk]
    model_used: str
