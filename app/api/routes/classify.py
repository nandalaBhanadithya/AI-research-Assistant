import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.constants import DOCUMENT_CATEGORIES
from app.models.chunk import Chunk
from app.schemas.classify import CategoriesResponse, ClassifyResponse
from app.services import classification_service, document_service
from app.services.embedding_service import embed_texts

router = APIRouter(prefix="/api/v1/classify", tags=["classify"])


@router.get("/categories", response_model=CategoriesResponse)
async def list_categories() -> CategoriesResponse:
    return CategoriesResponse(categories=DOCUMENT_CATEGORIES)


@router.post("/{document_id}", response_model=ClassifyResponse)
async def classify_document(document_id: str, db: Session = Depends(get_db)) -> ClassifyResponse:
    document = document_service.get_document(db, document_id)

    chunks = (
        db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).limit(2).all()
    )
    if chunks:
        lead_embeddings = await embed_texts([c.text for c in chunks])
        lead_vector = np.mean(lead_embeddings, axis=0).tolist()
        prediction = await classification_service.predict_category(lead_vector)
        if prediction:
            document.category, document.category_confidence = prediction
            db.commit()

    return ClassifyResponse(
        document_id=document_id, category=document.category, category_confidence=document.category_confidence
    )
