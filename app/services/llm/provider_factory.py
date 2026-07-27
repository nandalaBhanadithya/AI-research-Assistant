import logging
from functools import lru_cache
from typing import Optional

from app.config import get_settings
from app.core.exceptions import LLMProviderError
from app.services.llm.base import GenerationResult
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class FallbackGenerationProvider:
    """Wraps a primary generation provider; on failure, retries once via the fallback.

    Embeddings are never delegated here — get_embedding_provider() always returns the
    raw OllamaProvider directly, so this wrapper only ever affects generate().
    """

    def __init__(self, primary, fallback, enabled: bool):
        self._primary = primary
        self._fallback = fallback
        self._enabled = enabled
        self.name = primary.name

    async def generate(self, *args, **kwargs) -> GenerationResult:
        try:
            return await self._primary.generate(*args, **kwargs)
        except LLMProviderError as exc:
            if not self._enabled or self._primary is self._fallback:
                raise
            logger.warning("Primary provider %s failed (%s); falling back to %s", self._primary.name, exc, self._fallback.name)
            return await self._fallback.generate(*args, **kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._primary.embed(texts)

    def supports_embeddings(self) -> bool:
        return self._primary.supports_embeddings()


@lru_cache
def _ollama_provider() -> OllamaProvider:
    settings = get_settings()
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        generation_model=settings.ollama_generation_model,
        embedding_model=settings.ollama_embedding_model,
    )


@lru_cache
def _groq_provider() -> GroqProvider:
    settings = get_settings()
    return GroqProvider(base_url=settings.groq_base_url, api_key=settings.groq_api_key, model=settings.groq_model)


@lru_cache
def get_generation_provider() -> FallbackGenerationProvider:
    """Returns the active generation provider per GENERATION_PROVIDER, wrapped with
    fallback-to-local behavior if FALLBACK_TO_LOCAL_ON_ERROR is enabled."""
    settings = get_settings()
    primary = _groq_provider() if settings.generation_provider == "groq" else _ollama_provider()
    fallback = _ollama_provider()
    return FallbackGenerationProvider(primary, fallback, enabled=settings.fallback_to_local_on_error)


def get_embedding_provider() -> OllamaProvider:
    """Embeddings ALWAYS run locally via Ollama, regardless of GENERATION_PROVIDER."""
    return _ollama_provider()


async def check_provider_health() -> dict:
    settings = get_settings()
    ollama = _ollama_provider()
    groq = _groq_provider()
    return {
        "active_generation_provider": settings.generation_provider,
        "ollama_reachable": await ollama.is_reachable(),
        "groq_reachable": await groq.is_reachable() if settings.groq_api_key else None,
        "fallback_enabled": settings.fallback_to_local_on_error,
    }
