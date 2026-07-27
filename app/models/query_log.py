import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class QueryLog(Base):
    """Analytics source of truth: one row per search/qa/compare/summarize call."""

    __tablename__ = "query_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    endpoint: Mapped[str] = mapped_column(String(32))  # search | qa | compare | summarize
    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mode: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    insufficient_evidence: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
