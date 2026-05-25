"""SQLAlchemy models for stored posts (HTML is the source of truth for post content)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Post(Base):
    """One generated or edited post; caption/media details live in ``html_content``."""

    __tablename__ = "posts"

    class Status:
        COMPLETED = "completed"
        FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    post_name: Mapped[str] = mapped_column(String(512), default="")
    user_query: Mapped[str] = mapped_column(Text, default="")
    session_summary: Mapped[str] = mapped_column(Text, default="")
    html_content: Mapped[str] = mapped_column(Text, default="")
    cost_to_build_post: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        default=Decimal("0"),
    )

    parent_post_id: Mapped[int | None] = mapped_column(
        ForeignKey("posts.id"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(String(16), default=Status.COMPLETED)
    error_message: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class GenerationSession(Base):
    """Tracks in-flight SSE generation for cross-process cancel."""

    __tablename__ = "generation_sessions"

    class Status:
        RUNNING = "running"
        CANCEL_REQUESTED = "cancel_requested"
        FINISHED = "finished"
        CANCELLED = "cancelled"
        FAILED = "failed"

        TERMINAL = frozenset({FINISHED, CANCELLED, FAILED})

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default=Status.RUNNING, index=True)
    worker_id: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
