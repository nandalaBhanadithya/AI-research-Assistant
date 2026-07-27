import time
from typing import Optional

import httpx

from app.core.exceptions import LLMProviderError
from app.services.llm.base import GenerationResult


class OllamaProvider:
    """Local, project-scoped Ollama instance. Only provider allowed to serve embeddings."""

    name = "ollama"

    def __init__(self, base_url: str, generation_model: str, embedding_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> GenerationResult:
        # /api/chat (not /api/generate) — instruction-tuned models follow a system+user
        # message structure far more reliably than the raw prompt/system split that
        # /api/generate relies on the Modelfile template to reconstruct.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.generation_model,
            "messages": messages,
            "stream": False,
            # Ollama defaults num_ctx to 2048, which our retrieved-context prompts
            # regularly exceed — an overflowed context silently truncates/corrupts the
            # prompt rather than erroring, producing garbled output. 8192 comfortably
            # covers the system prompt + ~8 retrieved chunks + generation budget.
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
        }
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Ollama generation request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = data.get("message", {}).get("content", "")
        return GenerationResult(text=text, model_name=self.generation_model, latency_ms=latency_ms)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.embedding_model, "input": texts}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Ollama embedding request failed: {exc}") from exc

        return data["embeddings"]

    def supports_embeddings(self) -> bool:
        return True

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
