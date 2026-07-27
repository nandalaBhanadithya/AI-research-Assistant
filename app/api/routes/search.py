import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.query_log import QueryLog
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services import hybrid_search_service
from app.services.keyword_search_service import keyword_search

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    start = time.perf_counter()

    if payload.mode == "keyword":
        hits = keyword_search(db, payload.query, top_k=payload.top_k, document_ids=payload.document_ids)
        results = [
            SearchResultItem(
                chunk_id=h["chunk_id"],
                document_id=h["document_id"],
                document_name=h["document_name"],
                text=h["text"],
                page_start=h["page_start"],
                page_end=h["page_end"],
                section_label=h.get("section_label"),
                score=-h["bm25_rank"],
            )
            for h in hits
        ]
    elif payload.mode == "semantic":
        hits = await hybrid_search_service.semantic_search(
            payload.query, top_k=payload.top_k, document_ids=payload.document_ids
        )
        results = [
            SearchResultItem(
                chunk_id=h["chunk_id"],
                document_id=h["document_id"],
                document_name=h["document_name"],
                text=h["text"],
                page_start=h["page_start"],
                page_end=h["page_end"],
                section_label=h.get("section_label"),
                score=h["similarity"],
            )
            for h in hits
        ]
    else:
        hits = await hybrid_search_service.hybrid_search(
            db, payload.query, top_k=payload.top_k, document_ids=payload.document_ids
        )
        results = [
            SearchResultItem(
                chunk_id=h["chunk_id"],
                document_id=h["document_id"],
                document_name=h["document_name"],
                text=h["text"],
                page_start=h["page_start"],
                page_end=h["page_end"],
                section_label=h.get("section_label"),
                score=h["rrf_score"],
            )
            for h in hits
        ]

    db.add(
        QueryLog(
            endpoint="search",
            query_text=payload.query,
            mode=payload.mode,
            document_ids=list({r.document_id for r in results}),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    )
    db.commit()

    return SearchResponse(mode=payload.mode, query=payload.query, results=results)
