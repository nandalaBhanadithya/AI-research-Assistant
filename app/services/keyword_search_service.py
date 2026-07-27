import re
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

TOKEN_RE = re.compile(r"\w+")


def _build_fts_query(query: str) -> Optional[str]:
    """Builds a safe FTS5 MATCH expression from arbitrary user input. Tokens are
    individually quoted and OR'd so punctuation/operators in the raw query (hyphens,
    colons, unbalanced quotes) can never break FTS5's query syntax."""
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(
    db: Session, query: str, top_k: int = 8, document_ids: Optional[list[str]] = None
) -> list[dict]:
    fts_query = _build_fts_query(query)
    if fts_query is None:
        return []

    sql = """
        SELECT c.id AS chunk_id, c.document_id, d.name AS document_name, c.text, c.page_start,
               c.page_end, c.section_label, bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH :fts_query
    """
    params: dict = {"fts_query": fts_query, "top_k": top_k}
    if document_ids:
        placeholders = ", ".join(f":doc_{i}" for i in range(len(document_ids)))
        sql += f" AND c.document_id IN ({placeholders})"
        for i, doc_id in enumerate(document_ids):
            params[f"doc_{i}"] = doc_id
    sql += " ORDER BY rank LIMIT :top_k"

    rows = db.execute(text(sql), params).mappings().all()
    return [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "document_name": row["document_name"],
            "text": row["text"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "section_label": row["section_label"],
            "bm25_rank": row["rank"],
        }
        for row in rows
    ]
