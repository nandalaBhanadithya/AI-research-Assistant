from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.qa import CitationSchema


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2)
    aspects: Optional[list[str]] = None


class CompareResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    insufficient_evidence: bool
    confidence_score: float
    confidence_level: str
