"""Database access for posts."""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Post


def get_post(session: Session, pk: int) -> Post | None:
    return session.get(Post, pk)


def list_posts(
    session: Session,
    limit: int,
    order: Literal["asc", "desc"] = "desc",
) -> list[Post]:
    ord_col = Post.created_at.asc() if order == "asc" else Post.created_at.desc()
    stmt = select(Post).order_by(ord_col).limit(limit)
    return list(session.scalars(stmt))


def create_post(session: Session, **kwargs: Any) -> Post:
    parent = kwargs.pop("parent_post", None)
    if parent is not None:
        kwargs["parent_post_id"] = parent.id
    row = Post(**kwargs)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def recent_posts_memory(session: Session, limit: int = 5) -> str:
    stmt = (
        select(Post.post_name)
        .where(Post.status == Post.Status.COMPLETED)
        .order_by(Post.created_at.desc())
        .limit(limit)
    )
    lines: list[str] = []
    for (name,) in session.execute(stmt):
        n = (name or "").strip()
        if n:
            lines.append(f"- {n!r}")
    if not lines:
        return "(no prior completed posts in database)"
    return "Recent post names to avoid repeating themes:\n" + "\n".join(lines)
