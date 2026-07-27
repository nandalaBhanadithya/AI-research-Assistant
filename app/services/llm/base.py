from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class GenerationResult:
    text: str
    model_name: str
    latency_ms: int


class LLMProvider(Protocol):
    name: str

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> GenerationResult:
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def supports_embeddings(self) -> bool:
        ...
