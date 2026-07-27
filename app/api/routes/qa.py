from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.qa import QARequest, QAResponseSchema
from app.services import qa_service

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


@router.post("", response_model=QAResponseSchema)
async def ask_question(payload: QARequest, db: Session = Depends(get_db)) -> QAResponseSchema:
    result = await qa_service.answer_question(
        db, payload.question, conversation_id=payload.conversation_id, document_ids=payload.document_ids
    )
    return QAResponseSchema(
        answer=result.answer,
        citations=result.citations,
        confidence_score=result.confidence_score,
        confidence_level=result.confidence_level,
        insufficient_evidence=result.insufficient_evidence,
        conversation_id=result.conversation_id,
        retrieved_context=result.retrieved_context,
    )
