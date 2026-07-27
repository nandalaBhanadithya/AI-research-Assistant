import json
import re
from dataclasses import dataclass, field

from app.config import get_settings
from app.utils.text_utils import split_sentences

INSUFFICIENT_EVIDENCE_MESSAGE = "The answer could not be determined from the available documents."

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "and", "or", "for",
    "with", "as", "by", "that", "this", "it", "its", "be", "which", "at", "from", "these",
    "those", "such", "than", "into", "also", "can", "will", "not", "but",
}
_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


class OutputParseError(Exception):
    pass


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int


@dataclass
class VerifiedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False
    total_sentences: int = 0
    stripped_sentences: int = 0


def parse_json_object(raw_text: str) -> dict:
    """Generic JSON-object parser (strips markdown code fences some models add despite
    json_mode). Does not assume any particular schema — used directly by services whose
    schema has no 'answer' key (e.g. summarization)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputParseError(str(exc)) from exc
    if not isinstance(data, dict):
        raise OutputParseError(f"Expected a JSON object, got: {text[:200]!r}")
    return data


def parse_llm_json(raw_text: str) -> dict:
    """For the QA/compare answer schema specifically: parses JSON and additionally
    requires an 'answer' key. A small/weaker model can emit syntactically valid JSON
    that ignores our schema entirely (e.g. {"the whole answer as a key": [1]}) — that's
    treated as a parse failure too, so the caller's retry-with-stricter-prompt path
    kicks in rather than silently misreading the response."""
    data = parse_json_object(raw_text)
    if "answer" not in data:
        raise OutputParseError(f"JSON did not match expected schema (missing 'answer' key): {raw_text[:200]!r}")
    return data

    return data


def _significant_words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _overlap_ratio(sentence: str, reference_text: str) -> float:
    """Lexical (word-overlap) grounding check instead of a second embedding round-trip
    per sentence — keeps output verification at zero extra network latency, in line
    with the app's low-latency priority."""
    sentence_words = _significant_words(sentence)
    if not sentence_words:
        return 0.0
    reference_words = _significant_words(reference_text)
    return len(sentence_words & reference_words) / len(sentence_words)


def verify(raw_json: dict, retrieved_hits: list[dict]) -> VerifiedAnswer:
    settings = get_settings()
    hits_by_id = {h["chunk_id"]: h for h in retrieved_hits}

    answer_text = str(raw_json.get("answer", "")).strip()
    insufficient_evidence = bool(raw_json.get("insufficient_evidence", False))
    raw_citations = raw_json.get("citations") or []

    valid_citations: list[Citation] = []
    seen_chunk_ids: set = set()
    for raw_citation in raw_citations:
        chunk_id = raw_citation.get("chunk_id") if isinstance(raw_citation, dict) else None
        if chunk_id in hits_by_id and chunk_id not in seen_chunk_ids:
            hit = hits_by_id[chunk_id]
            valid_citations.append(
                Citation(
                    chunk_id=chunk_id,
                    document_id=hit["document_id"],
                    document_name=hit["document_name"],
                    page_start=hit["page_start"],
                    page_end=hit["page_end"],
                )
            )
            seen_chunk_ids.add(chunk_id)

    if insufficient_evidence or not answer_text:
        return VerifiedAnswer(answer=INSUFFICIENT_EVIDENCE_MESSAGE, insufficient_evidence=True)

    sentences = split_sentences(answer_text) or [answer_text]
    cited_text = " ".join(hits_by_id[c.chunk_id]["text"] for c in valid_citations)

    kept_sentences = []
    stripped_count = 0
    for sentence in sentences:
        if cited_text and _overlap_ratio(sentence, cited_text) >= settings.rag_verify_threshold:
            kept_sentences.append(sentence)
        else:
            stripped_count += 1

    if not kept_sentences:
        return VerifiedAnswer(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            insufficient_evidence=True,
            total_sentences=len(sentences),
            stripped_sentences=stripped_count,
        )

    return VerifiedAnswer(
        answer=" ".join(kept_sentences),
        citations=valid_citations,
        insufficient_evidence=False,
        total_sentences=len(sentences),
        stripped_sentences=stripped_count,
    )
