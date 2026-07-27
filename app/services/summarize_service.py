import logging

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, ProcessingError
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.guardrails.output_verifier import OutputParseError, parse_json_object
from app.services.llm.provider_factory import get_generation_provider

logger = logging.getLogger(__name__)

# Bounds how much document text goes into the summarization prompt so we never rely on
# providers correctly handling a very large context window. For documents that exceed
# this budget we evenly sample chunks across the whole document (always keeping the
# first/last) rather than truncating the tail — a full map-reduce summarization pass
# over all chunks is documented as a future improvement for very long documents.
SUMMARY_CHAR_BUDGET = 8000

SUMMARY_SYSTEM_PROMPT = (
    "You are a research assistant. Summarize the provided document CONTEXT ONLY — do not use "
    "outside knowledge. Respond as JSON using EXACTLY these keys: "
    '"executive_summary" (2-3 plain-language sentences for a non-technical audience), '
    '"technical_summary" (a paragraph covering methodology and results for a technical audience), '
    '"bullet_points" (a list of 5-8 concise bullet strings), and '
    '"key_takeaways" (a list of 3-5 short strings). Use no other keys.'
)


def _sample_chunks(chunks: list[Chunk], char_budget: int) -> list[Chunk]:
    if not chunks:
        return []
    if sum(c.char_count for c in chunks) <= char_budget:
        return chunks

    keep_count = max(3, char_budget // 1200)
    if keep_count >= len(chunks):
        return chunks
    indices = sorted({round(i * (len(chunks) - 1) / (keep_count - 1)) for i in range(keep_count)})
    return [chunks[i] for i in indices]


async def summarize_document(db: Session, document_id: str) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Document '{document_id}' not found")

    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    if not chunks:
        raise ProcessingError("Document has no processed chunks to summarize yet.")

    sampled = _sample_chunks(chunks, SUMMARY_CHAR_BUDGET)
    context_block = "\n\n".join(c.text for c in sampled)
    prompt = f"DOCUMENT: {document.name}\n\nCONTEXT:\n{context_block}\n\nProduce the four summaries described."

    provider = get_generation_provider()
    result = await provider.generate(prompt, system=SUMMARY_SYSTEM_PROMPT, temperature=0.2, max_tokens=1200, json_mode=True)

    data = _parse_with_retry(result.text)
    if data is None:
        retry_prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, no extra text before or after."
        result = await provider.generate(retry_prompt, system=SUMMARY_SYSTEM_PROMPT, temperature=0.0, max_tokens=1200, json_mode=True)
        data = _parse_with_retry(result.text) or {}

    return {
        "document_id": document_id,
        "document_name": document.name,
        "executive_summary": _as_text(data.get("executive_summary", "")),
        "technical_summary": _as_text(data.get("technical_summary", "")),
        "bullet_points": _as_text_list(data.get("bullet_points")),
        "key_takeaways": _as_text_list(data.get("key_takeaways")),
    }


def _as_text(value) -> str:
    """Some models wrap a single-string field in a list despite the schema instructions
    (e.g. executive_summary: ["..."]) — normalize rather than fail the whole request."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _as_text_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _parse_with_retry(raw_text: str):
    try:
        return parse_json_object(raw_text)
    except OutputParseError:
        return None
