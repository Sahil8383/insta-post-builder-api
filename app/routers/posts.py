"""Post generation API (streaming agent + slim post storage)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crud import get_post, list_posts, recent_posts_memory
from app.database import get_db, get_session_factory
from app.constants import USER_CANCEL_MESSAGE
from app.post_helpers import (
    AgentRunOutcome,
    _MISSING_QUERY,
    _parent_context_block,
    _post_json_full,
    _post_json_summary,
    _read_json_body,
    _resolve_parent_post,
    _resolve_query_and_hints,
    _sync_persist_stream_outcome,
)
from app.stream_registry import (
    apply_local_cancel_if_active,
    mark_finished,
    register,
    request_cancel,
    unregister,
)
from app.stream_sessions import (
    create_running_session,
    is_cancel_requested_in_db,
    mark_session_terminal,
    request_cancel_in_db,
)
from instagram.agent.orchestrator import run_post_agent
from instagram.agent.runner import sse_format_event, sse_heartbeat
from instagram.agent.usage_tracking import UsageLedger

router = APIRouter(tags=["posts"])
logger = logging.getLogger(__name__)

_stream_semaphore: asyncio.Semaphore | None = None


def _get_stream_semaphore() -> asyncio.Semaphore:
    global _stream_semaphore
    if _stream_semaphore is None:
        n = max(1, int(get_settings().max_concurrent_streams))
        _stream_semaphore = asyncio.Semaphore(n)
    return _stream_semaphore


class GenerateStreamBody(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    query: str = Field(default="", max_length=10000)
    topic: str = Field(default="", max_length=512)
    tone: str = Field(default="", max_length=128)
    target_audience: str = Field(default="", max_length=512)
    audience: str = Field(default="", max_length=512)
    media_mode: Literal["stock", "generate", "auto"] = "auto"
    post_id: int | None = Field(default=None, ge=1)
    post_name: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def query_or_hints(self) -> GenerateStreamBody:
        q = self.query.strip()
        if q:
            return self
        t = self.topic.strip()
        tn = self.tone.strip()
        aud = (self.target_audience or self.audience).strip()
        if t and tn and aud:
            return self
        raise ValueError(
            "Provide `query`, or all of `topic`, `tone`, and `target_audience`."
        )


class CancelStreamBody(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=64)


@router.post("/api/posts/generate/cancel")
async def post_generate_cancel(body: CancelStreamBody) -> JSONResponse:
    session_id = body.session_id.strip()
    local = await asyncio.to_thread(request_cancel, session_id)
    if local == "ok":
        await asyncio.to_thread(request_cancel_in_db, session_id)
        return JSONResponse({"ok": True})
    if local == "already_finished":
        raise HTTPException(status_code=409, detail="Session already finished.")

    db_result = await asyncio.to_thread(request_cancel_in_db, session_id)
    if db_result == "ok":
        return JSONResponse({"ok": True})
    if db_result == "already_finished":
        raise HTTPException(status_code=409, detail="Session already finished.")
    raise HTTPException(status_code=404, detail="Unknown or expired session.")


@router.post("/api/posts/generate/stream")
async def post_generate_stream(request: Request) -> Response:
    trace_id = str(uuid.uuid4())
    settings = get_settings()
    logger.info("[%s] post generate stream started", trace_id)

    raw = await _read_json_body(request)
    if isinstance(raw, JSONResponse):
        return raw
    try:
        body_model = GenerateStreamBody.model_validate(raw)
    except ValidationError as e:
        errs = e.errors()
        detail = str(errs[0].get("msg", "Invalid request body")) if errs else "Invalid request body"
        return JSONResponse({"detail": detail}, status_code=422)

    body = body_model.model_dump()
    query, topic, tone, audience = _resolve_query_and_hints(body)
    if query is None:
        return _MISSING_QUERY

    post_name_override = str(body.get("post_name") or "").strip()

    media_mode = str(body.get("media_mode") or "auto").strip().lower()
    if media_mode not in ("stock", "generate", "auto"):
        media_mode = "auto"

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        parent_post, err = _resolve_parent_post(db, body)
        if err:
            return err
        parent_context = _parent_context_block(parent_post) if parent_post else None
        memory = recent_posts_memory(db)
        parent_post_id = parent_post.id if parent_post else None
    finally:
        db.close()

    sem = _get_stream_semaphore()
    await sem.acquire()
    try:
        usage_ledger = UsageLedger()
        outcome = AgentRunOutcome(usage_ledger=usage_ledger)
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        q_max = max(1, int(settings.stream_queue_maxsize))
        event_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=q_max)

        await asyncio.to_thread(create_running_session, trace_id)
        register(
            trace_id,
            cancel_event=cancel_event,
            event_q=event_q,
            loop=loop,
            outcome=outcome,
        )

        async def _aq_put(item: tuple[str, Any]) -> None:
            await event_q.put(item)

        def threadsafe_put(item: tuple[str, Any]) -> None:
            fut = asyncio.run_coroutine_threadsafe(_aq_put(item), loop)
            fut.result()

        def emit(ev: dict[str, Any]) -> None:
            ev_out = dict(ev)
            ev_out.setdefault("trace_id", trace_id)
            threadsafe_put(("event", ev_out))

        def worker() -> None:
            logger.info("[%s] agent worker thread started", trace_id)
            try:
                outcome.result = run_post_agent(
                    query=query,
                    tone=tone,
                    target_audience=audience,
                    topic=topic,
                    parent_context=parent_context,
                    recent_posts_memory=memory,
                    media_mode=media_mode,
                    emit=emit,
                    usage_ledger=usage_ledger,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                outcome.error = exc
                if cancel_event.is_set():
                    outcome.cancelled = True
                if not outcome.user_cancel_notified:
                    msg = (
                        USER_CANCEL_MESSAGE
                        if outcome.cancelled
                        else str(exc)
                    )
                    emit({"type": "error", "message": msg})
                    outcome.worker_emitted_error = True
            finally:
                logger.info("[%s] agent worker thread exiting", trace_id)
                threadsafe_put(("done", None))

        worker_thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"agent-{trace_id[:8]}",
        )
        worker_thread.start()

        persist_lock = asyncio.Lock()

        async def ensure_persist() -> None:
            async with persist_lock:
                if outcome.persisted:
                    return
                logger.info("[%s] persist starting", trace_id)

                def join_worker() -> bool:
                    worker_thread.join(
                        timeout=float(settings.agent_thread_join_timeout_seconds)
                    )
                    return not worker_thread.is_alive()

                joined_ok = await asyncio.to_thread(join_worker)
                if not joined_ok:
                    logger.warning("[%s] worker join timed out", trace_id)
                    if outcome.error is None and outcome.result is None:
                        outcome.error = TimeoutError("Agent thread join timed out")

                try:
                    await asyncio.to_thread(
                        _sync_persist_stream_outcome,
                        SessionLocal,
                        outcome,
                        post_name_override=post_name_override,
                        query=query,
                        topic=topic,
                        parent_post_id=parent_post_id,
                        debug=settings.debug,
                    )
                except Exception as exc:
                    logger.exception("[%s] persist raised", trace_id)
                    outcome.persist_error = exc
                outcome.persisted = True
                logger.info(
                    "[%s] persist finished post_id=%s",
                    trace_id,
                    outcome.db_post_id,
                )

        get_timeout = float(settings.stream_get_timeout_seconds)
        cancel_poll_s = max(0.5, float(settings.stream_cancel_poll_seconds))
        wall_deadline = (
            time.monotonic()
            + float(settings.agent_max_runtime_seconds)
            + 120.0
        )

        async def event_stream():
            last_cancel_poll = 0.0
            try:
                yield "retry: 3000\n\n"
                eid = 0
                yield sse_format_event(
                    eid,
                    {"type": "stream-started", "session_id": trace_id},
                )
                eid += 1
                client_gone = False
                while True:
                    now = time.monotonic()
                    if now - last_cancel_poll >= cancel_poll_s:
                        last_cancel_poll = now
                        if await asyncio.to_thread(
                            is_cancel_requested_in_db, trace_id
                        ):
                            await asyncio.to_thread(
                                apply_local_cancel_if_active, trace_id
                            )

                    if await request.is_disconnected():
                        if not client_gone:
                            client_gone = True
                            cancel_event.set()
                            await asyncio.to_thread(request_cancel_in_db, trace_id)
                            logger.info("[%s] client disconnected", trace_id)

                    if time.monotonic() > wall_deadline:
                        if not cancel_event.is_set():
                            cancel_event.set()
                            logger.warning("[%s] stream wall deadline exceeded", trace_id)

                    try:
                        kind, payload = await asyncio.wait_for(
                            event_q.get(),
                            timeout=get_timeout,
                        )
                    except asyncio.TimeoutError:
                        yield sse_heartbeat()
                        continue

                    if kind == "done":
                        break

                    ev = payload
                    if client_gone:
                        continue
                    if ev.get("type") == "heartbeat":
                        yield sse_heartbeat()
                    else:
                        yield sse_format_event(eid, ev)
                        eid += 1

                await ensure_persist()

                if client_gone:
                    return

                if outcome.result is not None and outcome.error is None:
                    if outcome.db_post_id is None:
                        err_ev = {
                            "type": "error",
                            "message": "Persist did not return a post id",
                            "trace_id": trace_id,
                        }
                        yield sse_format_event(eid, err_ev)
                        eid += 1
                    else:
                        yield sse_format_event(
                            eid,
                            {
                                "type": "done",
                                "post_id": outcome.db_post_id,
                                "result_kind": outcome.db_result_kind or "post",
                                "session_summary": outcome.db_session_summary,
                                "trace_id": trace_id,
                            },
                        )
                        eid += 1
                else:
                    msg = (
                        USER_CANCEL_MESSAGE
                        if outcome.user_cancel_notified or outcome.cancelled
                        else (
                            str(outcome.error)
                            if outcome.error
                            else (
                                str(outcome.persist_error)
                                if outcome.persist_error
                                else "Unknown error"
                            )
                        )
                    )
                    err_ev = {
                        "type": "error",
                        "message": msg,
                        "trace_id": trace_id,
                    }
                    if outcome.db_post_id is not None:
                        err_ev["post_id"] = outcome.db_post_id
                    send_terminal_error = (
                        not outcome.user_cancel_notified
                        and (
                            outcome.db_post_id is not None
                            or not outcome.worker_emitted_error
                        )
                    )
                    if send_terminal_error:
                        yield sse_format_event(eid, err_ev)
                        eid += 1
            finally:
                mark_finished(trace_id)
                unregister(trace_id)
                if outcome.cancelled or outcome.user_cancel_notified:
                    terminal = "cancelled"
                elif outcome.result is not None and outcome.error is None:
                    terminal = "finished"
                else:
                    terminal = "failed"
                await asyncio.shield(
                    asyncio.to_thread(mark_session_terminal, trace_id, terminal)
                )
                await asyncio.shield(ensure_persist())
                sem.release()
                logger.info("[%s] stream generator finished", trace_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Stream-Session-Id": trace_id,
            },
        )
    except Exception:
        unregister(trace_id)
        try:
            await asyncio.to_thread(mark_session_terminal, trace_id, "failed")
        except Exception:
            logger.exception("[%s] failed to mark session terminal on setup error", trace_id)
        sem.release()
        raise


@router.get("/api/posts/")
def post_list(
    limit: int = 50,
    order: str = "desc",
    include_html: int = 0,
    db: Session = Depends(get_db),
) -> JSONResponse:
    lim = min(limit, 200)
    order_sql: Literal["asc", "desc"] = (
        "asc" if order.strip().lower() == "asc" else "desc"
    )
    rows = list_posts(db, lim, order=order_sql)
    want_html = include_html != 0
    if want_html:
        return JSONResponse({"results": [_post_json_full(pg) for pg in rows]})
    return JSONResponse({"results": [_post_json_summary(pg) for pg in rows]})


@router.get("/api/posts/{pk}/")
def post_detail(pk: int, db: Session = Depends(get_db)) -> JSONResponse:
    pg = get_post(db, pk)
    if pg is None:
        raise HTTPException(status_code=404, detail="Not found.")
    return JSONResponse(_post_json_full(pg))
