from typing import Optional

from pydantic import BaseModel


class ClassifyResponse(BaseModel):
    document_id: str
    category: Optional[str] = None
    category_confidence: Optional[float] = None


class CategoriesResponse(BaseModel):
    categories: list[str]
