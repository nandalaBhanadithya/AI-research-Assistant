import time
from typing import Optional

import httpx

from app.core.exceptions import LLMProviderError
from app.services.llm.base import GenerationResult


class GroqProvider:
    """Free-tier, low-latency generation via Groq's OpenAI-compatible API.

    Generation only — Groq has no embeddings endpoint, so embeddings always route
    through OllamaProvider regardless of which provider is generating answers.
    """

    name = "groq"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> GenerationResult:
        if not self.api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Groq generation request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = data["choices"][0]["message"]["content"]
        return GenerationResult(text=text, model_name=self.model, latency_ms=latency_ms)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Groq does not provide an embeddings API; use OllamaProvider for embeddings.")

    def supports_embeddings(self) -> bool:
        return False

    async def is_reachable(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
