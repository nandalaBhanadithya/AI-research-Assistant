import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.query_log import QueryLog
from app.schemas.compare import CompareRequest, CompareResponse
from app.services import compare_service

router = APIRouter(prefix="/api/v1/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
async def compare_documents(payload: CompareRequest, db: Session = Depends(get_db)) -> CompareResponse:
    start = time.perf_counter()
    result = await compare_service.compare_documents(db, payload.document_ids, aspects=payload.aspects)

    db.add(
        QueryLog(
            endpoint="compare",
            document_ids=payload.document_ids,
            confidence=result["confidence_score"],
            insufficient_evidence=result["insufficient_evidence"],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    )
    db.commit()

    return CompareResponse(**result)
