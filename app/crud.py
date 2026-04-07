"""Database access for post generations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PostGeneration, PostGenerationUsage
from instagram.agent.usage_tracking import UsageLedger


def get_post(session: Session, pk: int) -> PostGeneration | None:
    return session.get(PostGeneration, pk)


def get_post_usage(session: Session, post_id: int) -> PostGenerationUsage | None:
    stmt = select(PostGenerationUsage).where(PostGenerationUsage.post_id == post_id)
    return session.scalars(stmt).first()


def list_posts(session: Session, limit: int) -> list[PostGeneration]:
    stmt = (
        select(PostGeneration)
        .order_by(PostGeneration.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def create_post(session: Session, **kwargs: Any) -> PostGeneration:
    parent = kwargs.pop("parent_post", None)
    if parent is not None:
        kwargs["parent_post_id"] = parent.id
    row = PostGeneration(**kwargs)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def create_post_usage(session: Session, post_id: int, ledger: UsageLedger) -> PostGenerationUsage:
    row = PostGenerationUsage(
        post_id=post_id,
        usage_breakdown=ledger.to_breakdown_dict(),
        estimated_cost_usd=ledger.estimate_total_usd(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def recent_posts_memory(session: Session, limit: int = 5) -> str:
    stmt = (
        select(PostGeneration.topic, PostGeneration.caption, PostGeneration.post_type)
        .where(PostGeneration.status == PostGeneration.Status.COMPLETED)
        .order_by(PostGeneration.created_at.desc())
        .limit(limit)
    )
    lines: list[str] = []
    for topic, caption, post_type in session.execute(stmt):
        t = (topic or "").strip()
        cap = (caption or "").strip().replace("\n", " ")[:160]
        pt = (post_type or "").strip()
        if not cap and not t:
            continue
        extra = f" [{pt}]" if pt else ""
        lines.append(f"- topic={t!r}{extra} caption_preview={cap!r}")
    if not lines:
        return "(no prior completed posts in database)"
    return "Recent posts to avoid repeating hooks/topics:\n" + "\n".join(lines)
