import logging
from typing import Optional

import httpx

from app.core.exceptions import LLMProviderError
from app.services.llm.base import GenerationResult

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Hugging Face Inference API provider for embeddings."""

    name = "huggingface"

    def __init__(self, api_key: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.base_url = "https://api-inference.huggingface.co/models"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> GenerationResult:
        raise LLMProviderError("HuggingFaceProvider does not support text generation. Use it only for embeddings.")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Hugging Face Inference API."""
        if not texts:
            return []

        try:
            response = await self.client.post(
                f"{self.base_url}/{self.embedding_model}",
                json={"inputs": texts},
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle both single embedding and batch responses
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], list):
                    # Single text, single embedding
                    return [result[0]]
                elif all(isinstance(item, list) for item in result):
                    # Batch of embeddings
                    return result
                else:
                    # Single embedding returned as list of floats
                    return [result]
            elif isinstance(result, dict) and "embeddings" in result:
                return result["embeddings"]
            else:
                raise LLMProviderError(f"Unexpected response format: {type(result)}")
                
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(f"Hugging Face API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMProviderError(f"Failed to generate embeddings: {str(e)}")

    def supports_embeddings(self) -> bool:
        return True

    async def is_reachable(self) -> bool:
        """Check if the Hugging Face API is reachable."""
        try:
            response = await self.client.get(f"{self.base_url}/{self.embedding_model}")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
