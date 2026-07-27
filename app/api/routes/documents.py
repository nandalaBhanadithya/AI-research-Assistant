from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import ValidationFailedError
from app.models.chunk import Chunk
from app.schemas.document import ChunkResponse, DocumentListResponse, DocumentResponse
from app.services import document_service

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB per file


@router.post("", response_model=list[DocumentResponse], status_code=201)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> list[DocumentResponse]:
    created: list[DocumentResponse] = []
    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise ValidationFailedError(f"'{upload.filename}' is not a PDF file.")

        content = await upload.read()
        if not content:
            raise ValidationFailedError(f"'{upload.filename}' is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValidationFailedError(f"'{upload.filename}' exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.")

        document = document_service.create_document(db, upload.filename, content)
        background_tasks.add_task(document_service.process_document, document.id)
        created.append(DocumentResponse.model_validate(document))

    return created


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    documents, total = document_service.list_documents(db, status=status, category=category, limit=limit, offset=offset)
    return DocumentListResponse(total=total, documents=[DocumentResponse.model_validate(d) for d in documents])


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    document = document_service.get_document(db, document_id)
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    document_service.delete_document(db, document_id)


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> DocumentResponse:
    document = document_service.get_document(db, document_id)
    background_tasks.add_task(document_service.process_document, document.id)
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(document_id: str, db: Session = Depends(get_db)) -> list[ChunkResponse]:
    document_service.get_document(db, document_id)  # 404s if missing
    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    return [ChunkResponse.model_validate(c) for c in chunks]
