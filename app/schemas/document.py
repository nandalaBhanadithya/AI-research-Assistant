from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    upload_ts: datetime
    total_pages: int
    total_chunks: int
    processing_status: str
    error_message: Optional[str] = None
    category: Optional[str] = None
    category_confidence: Optional[float] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    page_start: int
    page_end: int
    section_label: Optional[str] = None
    token_count: int
    char_count: int
    text: str
