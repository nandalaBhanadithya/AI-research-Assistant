import logging
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.constants import ProcessingStatus
from app.core.exceptions import NotFoundError, ProcessingError
from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document
from app.services import chunking_service, classification_service, cleaning_service, embedding_service
from app.services import extraction_service, vector_store_service

logger = logging.getLogger(__name__)


def create_document(db: Session, filename: str, content: bytes) -> Document:
    settings = get_settings()
    document = Document(name=filename, file_path="", processing_status=ProcessingStatus.PENDING.value)
    db.add(document)
    db.flush()

    file_path = settings.upload_dir / f"{document.id}.pdf"
    file_path.write_bytes(content)
    document.file_path = str(file_path)
    db.commit()
    db.refresh(document)
    return document


def get_document(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Document '{document_id}' not found")
    return document


def list_documents(
    db: Session, status: Optional[str] = None, category: Optional[str] = None, limit: int = 50, offset: int = 0
) -> tuple[list[Document], int]:
    stmt = select(Document)
    if status:
        stmt = stmt.where(Document.processing_status == status)
    if category:
        stmt = stmt.where(Document.category == category)

    total = len(db.execute(stmt).scalars().all())
    stmt = stmt.order_by(Document.upload_ts.desc()).limit(limit).offset(offset)
    documents = list(db.execute(stmt).scalars().all())
    return documents, total


def delete_document(db: Session, document_id: str) -> None:
    document = get_document(db, document_id)
    vector_store_service.delete_document(document_id)
    try:
        settings = get_settings()
        file_path = settings.upload_dir / f"{document_id}.pdf"
        file_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove uploaded file for document %s", document_id, exc_info=True)
    db.delete(document)
    db.commit()


async def process_document(document_id: str) -> None:
    """Runs the full ingestion pipeline. Opens its own DB session so it works both as a
    FastAPI BackgroundTask (which outlives the request's session) and when invoked directly."""
    settings = get_settings()
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' not found")

        document.processing_status = ProcessingStatus.EXTRACTING.value
        db.commit()
        pages = extraction_service.extract_pdf(document.file_path)
        document.total_pages = len(pages)

        cleaned_pages = cleaning_service.clean_pages(pages)

        document.processing_status = ProcessingStatus.CHUNKING.value
        db.commit()
        drafts = chunking_service.chunk_document(cleaned_pages, settings.chunk_target_chars, settings.chunk_overlap_chars)
        if not drafts:
            raise ProcessingError(
                "No extractable text found in this document. Scanned/image-only PDFs are not "
                "supported in v1 (see README limitations — OCR is future work)."
            )

        document.processing_status = ProcessingStatus.EMBEDDING.value
        db.commit()
        embeddings = await embedding_service.embed_texts([d.text for d in drafts])

        db.query(Chunk).filter(Chunk.document_id == document.id).delete()
        db.flush()

        chunk_rows = [
            Chunk(
                document_id=document.id,
                chunk_index=d.chunk_index,
                page_start=d.page_start,
                page_end=d.page_end,
                section_label=d.section_label,
                text=d.text,
                token_count=d.token_count,
                char_count=d.char_count,
            )
            for d in drafts
        ]
        db.add_all(chunk_rows)
        db.flush()

        document.processing_status = ProcessingStatus.INDEXING.value
        db.commit()
        vector_store_service.delete_document(document.id)
        vector_store_service.add_chunks(
            chunk_ids=[c.id for c in chunk_rows],
            embeddings=embeddings,
            documents=[c.text for c in chunk_rows],
            metadatas=[
                {
                    "document_id": document.id,
                    "document_name": document.name,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_label": c.section_label or "",
                    "chunk_index": c.chunk_index,
                }
                for c in chunk_rows
            ],
        )
        document.total_chunks = len(chunk_rows)

        document.processing_status = ProcessingStatus.CLASSIFYING.value
        db.commit()
        lead_embeddings = embeddings[: min(2, len(embeddings))]
        if lead_embeddings:
            lead_vector = np.mean(lead_embeddings, axis=0).tolist()
            prediction = await classification_service.predict_category(lead_vector)
            if prediction:
                document.category, document.category_confidence = prediction

        document.processing_status = ProcessingStatus.COMPLETED.value
        document.error_message = None
        db.commit()
        logger.info("Document %s processed: %s pages, %s chunks", document_id, document.total_pages, document.total_chunks)

    except Exception as exc:
        db.rollback()
        document = db.get(Document, document_id)
        if document is not None:
            document.processing_status = ProcessingStatus.FAILED.value
            document.error_message = str(exc)[:2000]
            db.commit()
        logger.exception("Processing failed for document %s", document_id)
    finally:
        db.close()
