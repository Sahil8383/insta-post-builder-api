"""DB-backed generation session state for cross-worker cancel."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy.orm import Session

from app.database import get_session_factory
from app.models import GenerationSession
from app.stream_registry import CancelResult

CancelDbResult = CancelResult


def current_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_running_session(session_id: str, *, worker_id: str | None = None) -> None:
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        now = _now()
        row = GenerationSession(
            session_id=session_id,
            status=GenerationSession.Status.RUNNING,
            worker_id=worker_id or current_worker_id(),
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


def request_cancel_in_db(session_id: str) -> CancelDbResult:
    """Mark cancel requested; any worker running the stream should observe this."""
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        row = db.get(GenerationSession, session_id)
        if row is None:
            return "not_found"
        if row.status in GenerationSession.Status.TERMINAL:
            return "already_finished"
        if row.status == GenerationSession.Status.CANCEL_REQUESTED:
            return "ok"
        row.status = GenerationSession.Status.CANCEL_REQUESTED
        row.updated_at = _now()
        db.commit()
        return "ok"
    finally:
        db.close()

 
def is_cancel_requested_in_db(session_id: str) -> bool:
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        row = db.get(GenerationSession, session_id)
        return (
            row is not None
            and row.status == GenerationSession.Status.CANCEL_REQUESTED
        )
    finally:
        db.close()


def mark_session_terminal(
    session_id: str,
    status: Literal["finished", "cancelled", "failed"],
) -> None:
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        row = db.get(GenerationSession, session_id)
        if row is None:
            return
        if row.status in GenerationSession.Status.TERMINAL:
            return
        row.status = status
        row.updated_at = _now()
        db.commit()
    finally:
        db.close()
