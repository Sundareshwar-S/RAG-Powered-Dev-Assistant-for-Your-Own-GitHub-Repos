from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_URL: str = "http://ollama:11434"
    CHROMA_HOST: str = "http://chroma:8000"
    DEFAULT_LLM_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_NUM_CTX: int = 2048
    EMBED_BATCH_SIZE: int = 8
    EMBED_BACKEND: str = "ollama"
    EMBED_LOCAL_MODEL: str = "nomic-ai/nomic-embed-text-v1"
    EMBED_FAST_MODEL: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    EMBED_LOCAL_BATCH_SIZE: int = 64
    EMBED_FAST_BATCH_SIZE: int = 32
    EMBED_FAST_THREADS: int = 0
    EMBED_LOCAL_DEVICE: str = "cpu"
    EMBED_DIM: int = 768
    EMBED_KEEP_ALIVE: str = "5m"
    INGEST_FLUSH_SIZE: int = 64
    AST_MAX_SLIDING_CHUNKS: int = 12
    DOC_MAX_SLIDING_CHUNKS: int = 0
    HTML_MAX_SLIDING_CHUNKS: int = 3
    NOTEBOOK_MAX_CELLS: int = 50
    CHROMA_UPSERT_BATCH_SIZE: int = 128
    OLLAMA_CHAT_KEEP_ALIVE: str = "10m"
    OLLAMA_READ_TIMEOUT: float = 600.0
    SMALL_CORPUS_THRESHOLD: int = 50
    RETRIEVAL_TIMEOUT: float = 60.0
    DEBUG_TIMING: bool = False
    MAX_REPO_SIZE_MB: int = 500
    LOG_LEVEL: str = "info"
    BM25_CACHE_DIR: str = "/bm25_cache"
    GITHUB_TOKEN: str = ""


settings = Settings()
