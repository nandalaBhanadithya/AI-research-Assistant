from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.query_log import QueryLog
from app.services import vector_store_service


def overview(db: Session) -> dict:
    total_documents = db.scalar(select(func.count(Document.id))) or 0
    total_chunks = db.scalar(select(func.count(Chunk.id))) or 0
    total_embeddings = vector_store_service.count()
    total_questions_answered = db.scalar(select(func.count(QueryLog.id)).where(QueryLog.endpoint == "qa")) or 0

    documents_by_status = dict(
        db.execute(select(Document.processing_status, func.count(Document.id)).group_by(Document.processing_status)).all()
    )
    documents_by_category = dict(
        db.execute(
            select(Document.category, func.count(Document.id))
            .where(Document.category.isnot(None))
            .group_by(Document.category)
        ).all()
    )

    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_embeddings": total_embeddings,
        "total_questions_answered": total_questions_answered,
        "documents_by_status": documents_by_status,
        "documents_by_category": documents_by_category,
    }


def top_documents(db: Session, limit: int = 10) -> list[dict]:
    all_document_id_lists = db.execute(select(QueryLog.document_ids)).scalars().all()
    counter: Counter = Counter()
    for ids in all_document_id_lists:
        counter.update(ids or [])

    top = counter.most_common(limit)
    if not top:
        return []

    doc_ids = [doc_id for doc_id, _ in top]
    documents = {d.id: d for d in db.execute(select(Document).where(Document.id.in_(doc_ids))).scalars().all()}
    return [
        {
            "document_id": doc_id,
            "document_name": documents[doc_id].name if doc_id in documents else "(deleted)",
            "query_count": count,
        }
        for doc_id, count in top
    ]
