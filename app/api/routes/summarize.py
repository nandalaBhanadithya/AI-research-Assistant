import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.query_log import QueryLog
from app.schemas.summarize import SummarizeResponse
from app.services import summarize_service

router = APIRouter(prefix="/api/v1/documents", tags=["summarize"])


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
async def summarize_document(document_id: str, db: Session = Depends(get_db)) -> SummarizeResponse:
    start = time.perf_counter()
    result = await summarize_service.summarize_document(db, document_id)

    db.add(
        QueryLog(
            endpoint="summarize",
            document_ids=[document_id],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    )
    db.commit()

    return SummarizeResponse(**result)
