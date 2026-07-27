from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    sqlite_path: Path = PROJECT_ROOT / "data" / "sqlite" / "app.db"
    classifier_model_dir: Path = PROJECT_ROOT / "data" / "classifier" / "model"

    generation_provider: str = "groq"
    fallback_to_local_on_error: bool = True

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_generation_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"

    rag_top_k: int = 8
    rag_relevance_floor: float = 0.15
    rag_min_similarity: float = 0.35
    rag_verify_threshold: float = 0.25
    rag_max_hops: int = 2
    rag_max_hop_chunks: int = 12

    chunk_target_chars: int = 2000
    chunk_overlap_chars: int = 300

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.chroma_dir, self.sqlite_path.parent, self.classifier_model_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
