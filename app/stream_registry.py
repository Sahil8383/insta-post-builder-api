"""Local registry of active SSE streams plus cooperative cancel signaling."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Literal

from app.constants import USER_CANCEL_MESSAGE

CancelResult = Literal["ok", "not_found", "already_finished"]

_lock = threading.RLock()
_streams: dict[str, ActiveStream] = {}


@dataclass
class ActiveStream:
    session_id: str
    cancel_event: threading.Event
    event_q: asyncio.Queue[tuple[str, Any]]
    loop: asyncio.AbstractEventLoop
    outcome: Any
    finished: bool = False


def _apply_cancel_to_entry(entry: ActiveStream) -> None:
    entry.cancel_event.set()
    entry.outcome.cancelled = True
    entry.outcome.user_cancel_notified = True
    if entry.outcome.error is None:
        entry.outcome.error = RuntimeError(USER_CANCEL_MESSAGE)

    ev = {
        "type": "error",
        "message": USER_CANCEL_MESSAGE,
        "trace_id": entry.session_id,
    }

    async def _put() -> None:
        await entry.event_q.put(("event", ev))

    fut = asyncio.run_coroutine_threadsafe(_put(), entry.loop)
    fut.result(timeout=10.0)


def register(
    session_id: str,
    *,
    cancel_event: threading.Event,
    event_q: asyncio.Queue[tuple[str, Any]],
    loop: asyncio.AbstractEventLoop,
    outcome: Any,
) -> None:
    with _lock:
        _streams[session_id] = ActiveStream(
            session_id=session_id,
            cancel_event=cancel_event,
            event_q=event_q,
            loop=loop,
            outcome=outcome,
        )


def unregister(session_id: str) -> None:
    with _lock:
        entry = _streams.pop(session_id, None)
        if entry is not None:
            entry.finished = True


def mark_finished(session_id: str) -> None:
    with _lock:
        entry = _streams.get(session_id)
        if entry is not None:
            entry.finished = True


def request_cancel(session_id: str) -> CancelResult:
    """Cooperatively cancel a stream on this process and enqueue an SSE error."""
    with _lock:
        entry = _streams.get(session_id)
        if entry is None:
            return "not_found"
        if entry.finished:
            return "already_finished"
        _apply_cancel_to_entry(entry)
    return "ok"


def apply_local_cancel_if_active(session_id: str) -> bool:
    """Apply cancel when DB (or another path) requested it; returns True if applied."""
    with _lock:
        entry = _streams.get(session_id)
        if entry is None or entry.finished:
            return False
        if entry.cancel_event.is_set():
            return True
        _apply_cancel_to_entry(entry)
    return True
