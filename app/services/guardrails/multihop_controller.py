import json
import logging
import re
from typing import Optional

from app.config import get_settings
from app.services.hybrid_search_service import semantic_search
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_MULTIHOP_TRIGGER_RE = re.compile(
    r"\b(and (?:why|how|what|when|where|which)|as well as|in relation to|compared to|"
    r"and (?:its|their|his|her)|difference between|relationship between)\b",
    re.IGNORECASE,
)

DECOMPOSITION_SYSTEM_PROMPT = (
    "You break a single research question into at most 2 simpler, self-contained sub-questions "
    "that could each be answered by retrieving a specific passage from a document. "
    "Only decompose if the question genuinely has multiple parts. "
    "Respond ONLY as JSON: {\"sub_questions\": [\"...\", \"...\"]}. "
    "If the question is already simple/single-part, respond {\"sub_questions\": []}."
)


def looks_multihop(query: str) -> bool:
    return bool(_MULTIHOP_TRIGGER_RE.search(query)) or query.count("?") > 1


async def expand_context(
    query: str,
    hop1_hits: list[dict],
    generation_provider: LLMProvider,
    document_ids: Optional[list[str]] = None,
) -> tuple[list[dict], bool]:
    """Bounded multi-hop expansion: every 'hop' is a real retrieval call, never an
    invented reasoning step. Returns (merged_hits, multihop_was_used)."""
    settings = get_settings()
    if settings.rag_max_hops < 2 or not looks_multihop(query):
        return hop1_hits, False

    hop1_context = "\n\n".join(h["text"][:500] for h in hop1_hits[:4])
    prompt = f"Question: {query}\n\nInitial retrieved context (for grounding sub-questions only):\n{hop1_context}"

    try:
        result = await generation_provider.generate(
            prompt, system=DECOMPOSITION_SYSTEM_PROMPT, temperature=0.0, max_tokens=256, json_mode=True
        )
        sub_questions = json.loads(result.text).get("sub_questions", [])[:2]
    except Exception:
        logger.warning("Multi-hop decomposition failed; falling back to single-hop retrieval.", exc_info=True)
        return hop1_hits, False

    if not sub_questions:
        return hop1_hits, False

    merged: dict[str, dict] = {h["chunk_id"]: h for h in hop1_hits}
    hop2_k = max(2, settings.rag_top_k // 2)
    for sub_q in sub_questions:
        try:
            hop2_hits = await semantic_search(sub_q, top_k=hop2_k, document_ids=document_ids)
        except Exception:
            logger.warning("Hop-2 retrieval failed for sub-question %r", sub_q, exc_info=True)
            continue
        for hit in hop2_hits:
            existing = merged.get(hit["chunk_id"])
            if existing is None:
                hit["supports_sub_question"] = sub_q
                merged[hit["chunk_id"]] = hit

    ranked = sorted(merged.values(), key=lambda h: h["similarity"], reverse=True)
    return ranked[: settings.rag_max_hop_chunks], True
