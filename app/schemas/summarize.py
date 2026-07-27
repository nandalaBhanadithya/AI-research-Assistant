from typing import Optional

from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    types: Optional[list[str]] = None  # reserved for future selective summarization


class SummarizeResponse(BaseModel):
    document_id: str
    document_name: str
    executive_summary: str
    technical_summary: str
    bullet_points: list[str]
    key_takeaways: list[str]
