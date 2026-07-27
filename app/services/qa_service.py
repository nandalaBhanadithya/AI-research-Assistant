import time
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.core.constants import MessageRole
from app.models.query_log import QueryLog
from app.services import conversation_service
from app.services.guardrails import multihop_controller, output_verifier
from app.services.guardrails.confidence import compute_confidence
from app.services.guardrails.retrieval_gate import run_retrieval_gate
from app.services.llm.provider_factory import get_generation_provider

QA_SYSTEM_PROMPT = (
    "You are a research assistant that answers ONLY using the provided CONTEXT below. "
    "Do not use outside/general knowledge, even if you know the answer. "
    "If the context is insufficient to answer confidently, set insufficient_evidence=true "
    "and leave the answer empty. Every factual claim must be supported by a chunk_id present "
    "in CONTEXT — never invent a chunk_id that is not listed. "
    "You MUST respond with a JSON object using EXACTLY these three keys and no others: "
    '"answer" (a string), "citations" (a list of objects each with a "chunk_id" string), '
    'and "insufficient_evidence" (true or false). Do not use any other key names.\n'
    "Example of a correctly formatted response:\n"
    '{"answer": "The method uses residual connections to ease training of deep networks.", '
    '"citations": [{"chunk_id": "abc123"}], "insufficient_evidence": false}'
)

REFUSAL_MESSAGES = {
    "policy_violation": "I can't help with that request.",
    "off_topic": "That question doesn't appear to be covered by the uploaded documents.",
    "insufficient_evidence": "The answer could not be determined from the available documents.",
}


@dataclass
class QAResponse:
    answer: str
    citations: list[dict]
    confidence_score: float
    confidence_level: str
    insufficient_evidence: bool
    conversation_id: str
    retrieved_context: list[dict] = field(default_factory=list)


def _build_context_block(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[chunk_id={h['chunk_id']} document={h['document_name']} pages={h['page_start']}-{h['page_end']}]\n{h['text']}"
        for h in hits
    )


def _log_query(
    db: Session,
    question: str,
    conversation_id: str,
    document_ids: Optional[list[str]],
    insufficient: bool,
    start: float,
    confidence: Optional[float] = None,
) -> None:
    db.add(
        QueryLog(
            endpoint="qa",
            query_text=question,
            document_ids=document_ids or [],
            conversation_id=conversation_id,
            confidence=confidence,
            insufficient_evidence=insufficient,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    )
    db.commit()


def _refusal_response(
    db: Session, question: str, conversation_id: str, answer_text: str, intent_type: str, resolved_document_ids: list, start: float
) -> QAResponse:
    conversation_service.add_message(db, conversation_id, MessageRole.ASSISTANT.value, answer_text, intent_type=intent_type)
    _log_query(db, question, conversation_id, resolved_document_ids, insufficient=True, start=start)
    return QAResponse(
        answer=answer_text,
        citations=[],
        confidence_score=0.0,
        confidence_level="low",
        insufficient_evidence=True,
        conversation_id=conversation_id,
    )


async def answer_question(
    db: Session,
    question: str,
    conversation_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
) -> QAResponse:
    start = time.perf_counter()
    conversation = conversation_service.get_or_create_conversation(db, conversation_id)
    db.commit()
    conversation_id = conversation.id

    resolved_document_ids, needs_clarification = conversation_service.resolve_document_ids(
        db, question, conversation_id, document_ids
    )
    retrieval_query = conversation_service.build_retrieval_query(db, conversation_id, question)
    conversation_service.add_message(
        db, conversation_id, MessageRole.USER.value, question, referenced_document_ids=resolved_document_ids
    )

    if needs_clarification:
        return _refusal_response(
            db, question, conversation_id, "Which document are you asking about?", "clarification", resolved_document_ids, start
        )

    gate_result = await run_retrieval_gate(db, retrieval_query, document_ids=resolved_document_ids or None)
    if not gate_result.passed:
        answer_text = REFUSAL_MESSAGES.get(gate_result.reason, REFUSAL_MESSAGES["off_topic"])
        return _refusal_response(db, question, conversation_id, answer_text, gate_result.reason, resolved_document_ids, start)

    provider = get_generation_provider()
    hits, multihop_used = await multihop_controller.expand_context(
        question, gate_result.hits, provider, resolved_document_ids or None
    )

    prompt = f"CONTEXT:\n{_build_context_block(hits)}\n\nQUESTION: {question}"
    raw_result = await provider.generate(prompt, system=QA_SYSTEM_PROMPT, temperature=0.1, max_tokens=800, json_mode=True)

    try:
        raw_json = output_verifier.parse_llm_json(raw_result.text)
    except output_verifier.OutputParseError:
        retry_prompt = prompt + "\n\nReminder: respond with ONLY valid JSON, no extra text before or after."
        raw_result = await provider.generate(retry_prompt, system=QA_SYSTEM_PROMPT, temperature=0.0, max_tokens=800, json_mode=True)
        try:
            raw_json = output_verifier.parse_llm_json(raw_result.text)
        except output_verifier.OutputParseError:
            raw_json = {"answer": "", "citations": [], "insufficient_evidence": True}

    verified = output_verifier.verify(raw_json, hits)
    used_document_ids = list({c.document_id for c in verified.citations}) or resolved_document_ids

    confidence_score, confidence_level = compute_confidence(
        top_similarity=gate_result.top_similarity,
        total_sentences=verified.total_sentences,
        stripped_sentences=verified.stripped_sentences,
        num_citations=len(verified.citations),
        multihop_used=multihop_used,
    )

    conversation_service.add_message(
        db,
        conversation_id,
        MessageRole.ASSISTANT.value,
        verified.answer,
        referenced_document_ids=used_document_ids,
        intent_type="answer",
    )
    conversation_service.update_last_documents(db, conversation_id, used_document_ids)
    _log_query(db, question, conversation_id, used_document_ids, insufficient=verified.insufficient_evidence, start=start, confidence=confidence_score)

    return QAResponse(
        answer=verified.answer,
        citations=[vars(c) for c in verified.citations],
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        insufficient_evidence=verified.insufficient_evidence,
        conversation_id=conversation_id,
        retrieved_context=hits,
    )
