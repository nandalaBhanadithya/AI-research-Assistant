from fastapi import APIRouter

from app.services.llm.provider_factory import check_provider_health

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/llm")
async def health_llm() -> dict:
    return await check_provider_health()
