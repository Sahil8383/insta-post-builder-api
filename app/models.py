"""SQLAlchemy models (table post_generations, compatible with former Django schema)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.database import Base


class PostGeneration(Base):
    __tablename__ = "post_generations"

    class Status:
        COMPLETED = "completed"
        FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_query: Mapped[str] = mapped_column(Text, default="")
    intent: Mapped[str] = mapped_column(String(32), default="")
    topic: Mapped[str] = mapped_column(String(512), default="")
    tone: Mapped[str] = mapped_column(String(128), default="")
    target_audience: Mapped[str] = mapped_column(String(512), default="")

    caption: Mapped[str] = mapped_column(Text, default="")
    hashtags: Mapped[str] = mapped_column(Text, default="")
    search_notes: Mapped[str] = mapped_column(Text, default="")

    post_type: Mapped[str] = mapped_column(String(64), default="")
    overlay_text: Mapped[str] = mapped_column(String(256), default="")
    overlay_position: Mapped[str] = mapped_column(String(32), default="")
    text_style: Mapped[str] = mapped_column(String(32), default="")
    suggested_posting_time: Mapped[str] = mapped_column(String(64), default="")

    image_url: Mapped[str] = mapped_column(String(2048), default="")
    video_url: Mapped[str] = mapped_column(String(2048), default="")
    media_type: Mapped[str] = mapped_column(String(32), default="image")
    media_attribution: Mapped[str] = mapped_column(Text, default="")
    image_prompt: Mapped[str] = mapped_column(Text, default="")

    insights_summary: Mapped[str] = mapped_column(Text, default="")
    insights_bullets: Mapped[list[Any]] = mapped_column(JSON, default=lambda: [])

    engagement_package: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=lambda: {},
    )

    session_summary: Mapped[str] = mapped_column(Text, default="")

    parent_post_id: Mapped[int | None] = mapped_column(
        ForeignKey("post_generations.id"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(String(16), default=Status.COMPLETED)
    error_message: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class PostGenerationUsage(Base):
    """Billable usage for one post_generation row (1:1)."""

    __tablename__ = "post_generation_usage"
    __table_args__ = (UniqueConstraint("post_id", name="uq_post_generation_usage_post_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(
        ForeignKey("post_generations.id", ondelete="CASCADE"),
        nullable=False,
    )
    usage_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=lambda: {})
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        default=Decimal("0"),
    )
