import re
from typing import Optional

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from app.core.constants import MessageRole
from app.models.conversation import Conversation, ConversationState, Message
from app.models.document import Document

REFERRING_EXPRESSION_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|their|the document|the paper|the other one|"
    r"the first (?:paper|document|one)|the second (?:paper|document|one))\b",
    re.IGNORECASE,
)

FUZZY_MATCH_THRESHOLD = 80
MAX_TRACKED_DOCUMENTS = 3


def get_or_create_conversation(db: Session, conversation_id: Optional[str]) -> Conversation:
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation is not None:
            return conversation
    conversation = Conversation()
    db.add(conversation)
    db.flush()
    return conversation


def _fuzzy_match_document_names(db: Session, query: str) -> list[str]:
    documents = db.query(Document).all()
    scored = []
    for doc in documents:
        score = fuzz.partial_ratio(query.lower(), doc.name.lower())
        if score >= FUZZY_MATCH_THRESHOLD:
            scored.append((score, doc.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc_id for _, doc_id in scored]


def resolve_document_ids(
    db: Session,
    query: str,
    conversation_id: Optional[str],
    explicit_document_ids: Optional[list[str]],
) -> tuple[list[str], bool]:
    """Returns (resolved_document_ids, needs_clarification).

    Resolution order: explicit doc_ids/filename match always wins > anaphoric reference
    or an ongoing conversation resolves against the conversation's last-discussed docs >
    otherwise unfiltered (search across all documents).
    """
    if explicit_document_ids:
        return explicit_document_ids, False

    fuzzy_matches = _fuzzy_match_document_names(db, query)
    if fuzzy_matches:
        return fuzzy_matches, False

    last_document_ids: list[str] = []
    if conversation_id:
        state = db.get(ConversationState, conversation_id)
        last_document_ids = list(state.last_document_ids) if state else []

    is_anaphoric = bool(REFERRING_EXPRESSION_RE.search(query))
    if last_document_ids:
        return last_document_ids, False

    if is_anaphoric:
        # No conversation history to resolve against — but if exactly one document
        # exists at all, "it"/"this paper" is unambiguous even on a fresh conversation.
        all_doc_ids = [doc.id for doc in db.query(Document.id).all()]
        if len(all_doc_ids) == 1:
            return all_doc_ids, False
        return [], True

    return [], False


def update_last_documents(db: Session, conversation_id: str, used_document_ids: list[str]) -> None:
    if not used_document_ids:
        return
    state = db.get(ConversationState, conversation_id)
    if state is None:
        state = ConversationState(conversation_id=conversation_id, last_document_ids=[])
        db.add(state)

    merged = list(dict.fromkeys([*used_document_ids, *state.last_document_ids]))
    state.last_document_ids = merged[:MAX_TRACKED_DOCUMENTS]
    db.commit()


def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    referenced_document_ids: Optional[list[str]] = None,
    intent_type: Optional[str] = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        referenced_document_ids=referenced_document_ids or [],
        intent_type=intent_type,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_history(db: Session, conversation_id: str, limit: int = 6) -> list[Message]:
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(messages))


def build_retrieval_query(db: Session, conversation_id: Optional[str], question: str) -> str:
    """Short/anaphoric follow-ups ("What are its limitations?") embed poorly on their
    own — a bare pronoun gives the embedding model nothing to anchor to, so it ranks
    fairly uniformly across an entire document. Prepending the previous user turn gives
    concrete anchoring (e.g. the paper/topic name) without changing what's shown to the
    LLM, which still sees the original question text."""
    is_short_or_anaphoric = bool(REFERRING_EXPRESSION_RE.search(question)) or len(question.split()) <= 6
    if not is_short_or_anaphoric or not conversation_id:
        return question

    history = get_history(db, conversation_id, limit=4)
    previous_user_messages = [m.content for m in history if m.role == MessageRole.USER.value]
    if not previous_user_messages:
        return question
    return f"{previous_user_messages[-1]} {question}"
