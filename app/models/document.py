import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import ProcessingStatus
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    name: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    upload_ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    processing_status: Mapped[str] = mapped_column(String(32), default=ProcessingStatus.PENDING.value)
    error_message: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    category_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )
