from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(keyword|semantic|hybrid)$")
    document_ids: Optional[list[str]] = None
    top_k: int = Field(default=8, ge=1, le=50)


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    section_label: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    mode: str
    query: str
    results: list[SearchResultItem]
