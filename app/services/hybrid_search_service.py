from typing import Optional

from app.services import embedding_service, vector_store_service
from app.services.keyword_search_service import keyword_search

RRF_K = 60


async def semantic_search(query: str, top_k: int = 8, document_ids: Optional[list[str]] = None) -> list[dict]:
    query_embedding = (await embedding_service.embed_texts([query]))[0]
    hits = vector_store_service.query(query_embedding, top_k=top_k, document_ids=document_ids)
    return [
        {
            "chunk_id": h["chunk_id"],
            "document_id": h["metadata"]["document_id"],
            "document_name": h["metadata"]["document_name"],
            "text": h["text"],
            "page_start": h["metadata"]["page_start"],
            "page_end": h["metadata"]["page_end"],
            "section_label": h["metadata"].get("section_label") or None,
            "similarity": h["similarity"],
        }
        for h in hits
    ]


async def hybrid_search(
    db, query: str, top_k: int = 8, document_ids: Optional[list[str]] = None
) -> list[dict]:
    """Reciprocal Rank Fusion over keyword (FTS5/bm25) and semantic (Chroma cosine)
    result lists. Rank-based fusion sidesteps the fact that bm25 and cosine scores
    live on incompatible scales — no score normalization/tuning required."""
    keyword_hits = keyword_search(db, query, top_k=top_k * 2, document_ids=document_ids)
    semantic_hits = await semantic_search(query, top_k=top_k * 2, document_ids=document_ids)

    fused: dict[str, dict] = {}
    for rank, hit in enumerate(keyword_hits):
        entry = fused.setdefault(hit["chunk_id"], {**hit, "rrf_score": 0.0})
        entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
    for rank, hit in enumerate(semantic_hits):
        entry = fused.setdefault(hit["chunk_id"], {**hit, "rrf_score": 0.0})
        entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        if "similarity" in hit:
            entry["similarity"] = hit["similarity"]

    ranked = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    return ranked[:top_k]
