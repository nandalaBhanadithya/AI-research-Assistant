from typing import Optional

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: Optional[str] = None
    document_ids: Optional[list[str]] = None


class CitationSchema(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    similarity: Optional[float] = None


class QAResponseSchema(BaseModel):
    answer: str
    citations: list[CitationSchema]
    confidence_score: float
    confidence_level: str
    insufficient_evidence: bool
    conversation_id: str
    retrieved_context: list[RetrievedChunkSchema] = []
