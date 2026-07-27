from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/overview")
async def overview(db: Session = Depends(get_db)) -> dict:
    return analytics_service.overview(db)


@router.get("/top-documents")
async def top_documents(limit: int = 10, db: Session = Depends(get_db)) -> list[dict]:
    return analytics_service.top_documents(db, limit=limit)
