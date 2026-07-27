from app.services.llm.provider_factory import get_embedding_provider

EMBED_BATCH_SIZE = 32


async def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []
    provider = get_embedding_provider()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(await provider.embed(batch))
    return vectors
