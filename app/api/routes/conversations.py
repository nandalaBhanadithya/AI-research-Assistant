from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.conversation import Conversation
from app.schemas.conversation import ConversationDetail, ConversationSummary

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.post("", response_model=ConversationSummary, status_code=201)
async def create_conversation(db: Session = Depends(get_db)) -> ConversationSummary:
    conversation = Conversation()
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return ConversationSummary.model_validate(conversation)


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(db: Session = Depends(get_db)) -> list[ConversationSummary]:
    conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return [ConversationSummary.model_validate(c) for c in conversations]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationDetail:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation '{conversation_id}' not found")
    return ConversationDetail.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation '{conversation_id}' not found")
    db.delete(conversation)
    db.commit()
