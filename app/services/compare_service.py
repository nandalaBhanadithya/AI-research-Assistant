from typing import Optional

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, ValidationFailedError
from app.models.document import Document
from app.services.guardrails import output_verifier
from app.services.guardrails.confidence import compute_confidence
from app.services.hybrid_search_service import semantic_search
from app.services.llm.provider_factory import get_generation_provider

DEFAULT_ASPECTS = [
    "methodology",
    "advantages and disadvantages",
    "similarities",
    "differences",
    "conclusions",
    "implementation approach",
]

COMPARE_SYSTEM_PROMPT = (
    "You are a research assistant comparing multiple documents. Use ONLY the provided CONTEXT, "
    "which is tagged by document. Do not use outside knowledge. Organize your comparison by aspect. "
    "You MUST respond with JSON using EXACTLY these keys: \"answer\" (a string comparing the documents, "
    "organized by aspect), \"citations\" (a list of objects each with a \"chunk_id\" string), and "
    "\"insufficient_evidence\" (true or false). Every claim must cite a chunk_id present in CONTEXT."
)


async def compare_documents(db: Session, document_ids: list[str], aspects: Optional[list[str]] = None) -> dict:
    if len(document_ids) < 2:
        raise ValidationFailedError("At least 2 document_ids are required for comparison.")

    documents = {d.id: d for d in db.query(Document).filter(Document.id.in_(document_ids)).all()}
    missing = [doc_id for doc_id in document_ids if doc_id not in documents]
    if missing:
        raise NotFoundError(f"Document(s) not found: {', '.join(missing)}")

    aspects = aspects or DEFAULT_ASPECTS
    query = "Key points about: " + ", ".join(aspects)

    all_hits: list[dict] = []
    for doc_id in document_ids:
        all_hits.extend(await semantic_search(query, top_k=6, document_ids=[doc_id]))

    if not all_hits:
        return {
            "answer": "Insufficient retrievable content in the selected documents to perform a comparison.",
            "citations": [],
            "insufficient_evidence": True,
            "confidence_score": 0.0,
            "confidence_level": "low",
        }

    context_lines = []
    for doc_id in document_ids:
        doc_hits = [h for h in all_hits if h["document_id"] == doc_id]
        context_lines.append(f"=== Document: {documents[doc_id].name} (document_id={doc_id}) ===")
        context_lines.extend(f"[chunk_id={h['chunk_id']} pages={h['page_start']}-{h['page_end']}]\n{h['text']}" for h in doc_hits)
    context_block = "\n\n".join(context_lines)

    prompt = f"CONTEXT:\n{context_block}\n\nCompare these documents across: {', '.join(aspects)}."

    provider = get_generation_provider()
    result = await provider.generate(prompt, system=COMPARE_SYSTEM_PROMPT, temperature=0.2, max_tokens=1200, json_mode=True)

    try:
        raw_json = output_verifier.parse_llm_json(result.text)
    except output_verifier.OutputParseError:
        retry_prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, no extra text before or after."
        result = await provider.generate(retry_prompt, system=COMPARE_SYSTEM_PROMPT, temperature=0.0, max_tokens=1200, json_mode=True)
        try:
            raw_json = output_verifier.parse_llm_json(result.text)
        except output_verifier.OutputParseError:
            raw_json = {"answer": "", "citations": [], "insufficient_evidence": True}

    verified = output_verifier.verify(raw_json, all_hits)
    top_similarity = max((h["similarity"] for h in all_hits), default=0.0)
    confidence_score, confidence_level = compute_confidence(
        top_similarity=top_similarity,
        total_sentences=verified.total_sentences,
        stripped_sentences=verified.stripped_sentences,
        num_citations=len(verified.citations),
        multihop_used=False,
    )

    return {
        "answer": verified.answer,
        "citations": [vars(c) for c in verified.citations],
        "insufficient_evidence": verified.insufficient_evidence,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
    }
