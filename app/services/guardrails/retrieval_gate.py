import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.services.hybrid_search_service import RRF_K, semantic_search
from app.services.keyword_search_service import keyword_search

# A deliberately small deny-list rather than an ML classifier — cheap, deterministic,
# and catches the common "ignore your instructions" style prompt-injection attempts.
_INJECTION_RE = re.compile(
    r"ignore (?:all |the )?(?:previous|prior|above) instructions|"
    r"disregard (?:the )?system prompt|reveal (?:your |the )?(?:system )?prompt|"
    r"you are now|act as (?:a|an) unrestricted|jailbreak",
    re.IGNORECASE,
)


@dataclass
class GateResult:
    passed: bool
    reason: Optional[str]  # None | "policy_violation" | "off_topic" | "insufficient_evidence"
    hits: list[dict] = field(default_factory=list)
    top_similarity: float = 0.0


def _looks_like_injection(query: str) -> bool:
    return bool(_INJECTION_RE.search(query))


def _merge_with_keyword_recall(semantic_hits: list[dict], keyword_hits: list[dict]) -> list[dict]:
    """RRF-merges keyword hits into the semantic result set so short/ambiguous queries
    (e.g. 'what are its limitations') still surface passages a keyword match would catch
    even when their embedding similarity alone isn't top-ranked. Only run after the gate
    has already passed, so a refused query never pays for the extra FTS5 lookup."""
    merged: dict[str, dict] = {h["chunk_id"]: {**h, "similarity": h["similarity"]} for h in semantic_hits}
    for rank, hit in enumerate(keyword_hits):
        if hit["chunk_id"] not in merged:
            hit = dict(hit)
            hit["similarity"] = 1.0 / (RRF_K + rank + 1)  # keyword-only hit: small nonzero relevance signal
            merged[hit["chunk_id"]] = hit
    return list(merged.values())


async def run_retrieval_gate(db, query: str, document_ids: Optional[list[str]] = None) -> GateResult:
    """Gates on pure cosine similarity (relevance-floor / grounding-threshold constants
    are calibrated in similarity units, not rank-fusion scores) so a refused/off-topic
    query never pays for more than one retrieval call. Once the gate passes, keyword
    (FTS5) recall is merged in for the context actually sent to the LLM — semantic-only
    retrieval regularly missed exact-phrase passages (e.g. "degradation problem") for
    short paraphrased questions like "what are its limitations".
    """
    if _looks_like_injection(query):
        return GateResult(passed=False, reason="policy_violation")

    settings = get_settings()
    semantic_hits = await semantic_search(query, top_k=settings.rag_top_k, document_ids=document_ids)
    top_similarity = semantic_hits[0]["similarity"] if semantic_hits else 0.0

    if not semantic_hits or top_similarity < settings.rag_relevance_floor:
        return GateResult(passed=False, reason="off_topic", hits=semantic_hits, top_similarity=top_similarity)

    if top_similarity < settings.rag_min_similarity:
        return GateResult(passed=False, reason="insufficient_evidence", hits=semantic_hits, top_similarity=top_similarity)

    keyword_hits = keyword_search(db, query, top_k=settings.rag_top_k, document_ids=document_ids)
    merged_hits = _merge_with_keyword_recall(semantic_hits, keyword_hits)
    return GateResult(passed=True, reason=None, hits=merged_hits, top_similarity=top_similarity)
