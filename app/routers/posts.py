"""Post generation API (streaming agent + slim post storage)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crud import create_post, get_post, list_posts, recent_posts_memory
from app.database import get_db, get_session_factory
from app.models import Post
from instagram.agent.orchestrator import agent_result_to_persist_content, run_post_agent
from instagram.agent.runner import AgentResult, sse_format_event, sse_heartbeat
from instagram.agent.usage_tracking import UsageLedger

router = APIRouter(tags=["posts"])
logger = logging.getLogger(__name__)

USER_QUERY_MAX_LEN = 16_000
SESSION_SUMMARY_MAX_LEN = 8_000

_stream_semaphore: asyncio.Semaphore | None = None


def _truncate_user_query(q: str) -> str:
    q = (q or "").strip()
    if len(q) <= USER_QUERY_MAX_LEN:
        return q
    return q[:USER_QUERY_MAX_LEN] + "\n…"


def _truncate_session_summary(s: str) -> str:
    s = (s or "").strip()
    if len(s) <= SESSION_SUMMARY_MAX_LEN:
        return s
    return s[:SESSION_SUMMARY_MAX_LEN] + "\n…"


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


@dataclass
class AgentRunOutcome:
    """Filled by the worker thread; DB fields filled by persist (same process)."""

    usage_ledger: UsageLedger
    result: AgentResult | None = None
    error: Exception | None = None
    cancelled: bool = False
    worker_emitted_error: bool = False
    persisted: bool = False
    db_post_id: int | None = None
    db_result_kind: str | None = None
    db_session_summary: str = ""
    persist_error: Exception | None = None


_MISSING_QUERY = JSONResponse(
    {
        "detail": "Provide `query`, or all of `topic`, `tone`, and `target_audience`.",
    },
    status_code=400,
)


def _debug_error_text(exc: BaseException, debug: bool) -> str:
    if debug:
        return f"{exc!s}\n{traceback.format_exc()}"
    return str(exc)


def _post_json_summary(pg: Post) -> dict[str, Any]:
    """List / metadata: no ``html_content`` (large)."""
    out: dict[str, Any] = {
        "id": pg.id,
        "post_name": pg.post_name,
        "user_query": pg.user_query or "",
        "session_summary": pg.session_summary or "",
        "cost_to_build_post": float(pg.cost_to_build_post),
        "status": pg.status,
        "created_at": pg.created_at.isoformat(),
    }
    if pg.parent_post_id is not None:
        out["parent_post_id"] = pg.parent_post_id
    err = (pg.error_message or "").strip()
    if err:
        out["error_message"] = err
    return out


def _post_json_full(pg: Post) -> dict[str, Any]:
    """Detail: includes ``html_content`` for iframe preview."""
    out = _post_json_summary(pg)
    out["html_content"] = pg.html_content
    return out


def _parent_context_block(pg: Post) -> str:
    html = (pg.html_content or "").strip()
    max_ctx = 120_000
    if len(html) > max_ctx:
        html = html[:max_ctx] + "\n<!-- truncated for agent context -->"
    return (
        "Current post HTML (edit in place when updating; keep structure where helpful):\n"
        + html
    )


def _resolve_query_and_hints(
    body: dict[str, Any],
) -> tuple[str, str, str, str] | tuple[None, None, None, None]:
    query = (body.get("query") or "").strip()
    topic = (body.get("topic") or "").strip()
    tone = (body.get("tone") or "").strip()
    audience = (body.get("target_audience") or body.get("audience") or "").strip()

    if not query:
        if topic and tone and audience:
            query = (
                "Create a complete Instagram post package.\n"
                f"topic: {topic}\ntone: {tone}\ntarget_audience: {audience}"
            )
        else:
            return None, None, None, None
    return query, topic, tone, audience


def _resolve_parent_post(
    session: Session, body: dict[str, Any]
) -> tuple[Post | None, JSONResponse | None]:
    raw = body.get("post_id")
    if raw is None or raw == "":
        return None, None
    try:
        pk = int(raw)
    except (TypeError, ValueError):
        return None, JSONResponse(
            {"detail": "post_id must be an integer."}, status_code=400
        )
    parent = get_post(session, pk)
    if parent is None:
        return None, JSONResponse({"detail": "post_id not found."}, status_code=404)
    return parent, None


def _row_topic_fallback(topic: str, query: str) -> str:
    t = (topic or "").strip()
    if t:
        return t[:512]
    q = query.strip().replace("\n", " ")
    return (q[:512] if q else "Untitled")


def _failed_post_display_name(post_name_override: str, topic: str, query: str) -> str:
    if (post_name_override or "").strip():
        return post_name_override.strip()[:512]
    return _row_topic_fallback(topic, query)


def _failed_row_kwargs(
    *,
    post_name_override: str,
    topic: str,
    query: str,
    err_text: str,
    cost: Decimal,
    parent_post_id: int | None = None,
    session_summary: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "post_name": _failed_post_display_name(post_name_override, topic, query),
        "user_query": _truncate_user_query(query),
        "session_summary": _truncate_session_summary(session_summary),
        "html_content": "",
        "cost_to_build_post": cost,
        "status": Post.Status.FAILED,
        "error_message": err_text,
    }
    if parent_post_id is not None:
        row["parent_post_id"] = parent_post_id
    return row


def _persist_agent_success(
    session: Session,
    *,
    post_name_override: str,
    parent_post_id: int | None,
    agent_result: Any,
    usage_ledger: UsageLedger,
    user_query: str,
) -> tuple[Post, str, str]:
    content = agent_result_to_persist_content(agent_result)
    rk = content["result_kind"]
    session_summary = content["session_summary"]
    name = (post_name_override.strip() or content["suggested_post_name"])[:512]
    cost = usage_ledger.estimate_total_usd()
    kwargs: dict[str, Any] = {
        "post_name": name,
        "user_query": _truncate_user_query(user_query),
        "session_summary": _truncate_session_summary(session_summary),
        "html_content": content["html_content"],
        "cost_to_build_post": cost,
        "status": Post.Status.COMPLETED,
    }
    if parent_post_id is not None:
        kwargs["parent_post_id"] = parent_post_id
    pg = create_post(session, **kwargs)
    return pg, rk, session_summary


def _sync_persist_stream_outcome(
    SessionLocal: Any,
    outcome: AgentRunOutcome,
    *,
    post_name_override: str,
    query: str,
    topic: str,
    parent_post_id: int | None,
    debug: bool,
) -> None:
    """Write success or failure row; update outcome with DB ids."""
    session: Session = SessionLocal()
    try:
        err = outcome.error
        cost = outcome.usage_ledger.estimate_total_usd()
        if err is not None:
            msg = (
                "Request cancelled"
                if outcome.cancelled
                else _debug_error_text(err, debug)
            )
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    post_name_override=post_name_override,
                    topic=topic,
                    query=query,
                    err_text=msg,
                    cost=cost,
                    parent_post_id=parent_post_id,
                ),
            )
            outcome.db_post_id = pg.id
            return

        if outcome.result is None:
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    post_name_override=post_name_override,
                    topic=topic,
                    query=query,
                    err_text="No agent result",
                    cost=cost,
                    parent_post_id=parent_post_id,
                ),
            )
            outcome.db_post_id = pg.id
            return

        try:
            pg, rk, summ = _persist_agent_success(
                session,
                post_name_override=post_name_override,
                parent_post_id=parent_post_id,
                agent_result=outcome.result,
                usage_ledger=outcome.usage_ledger,
                user_query=query,
            )
            outcome.db_post_id = pg.id
            outcome.db_result_kind = rk
            outcome.db_session_summary = summ
        except Exception as exc:
            outcome.persist_error = exc
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    post_name_override=post_name_override,
                    topic=topic,
                    query=query,
                    err_text=_debug_error_text(exc, debug),
                    cost=cost,
                    parent_post_id=parent_post_id,
                ),
            )
            outcome.db_post_id = pg.id
    finally:
        session.close()


async def _read_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    raw = await request.body()
    settings = get_settings()
    max_b = int(settings.max_stream_request_body_bytes)
    if len(raw) > max_b:
        return JSONResponse(
            {"detail": "Request body too large."},
            status_code=413,
        )
    if not raw.strip():
        return {}
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            return JSONResponse({"detail": "Invalid JSON"}, status_code=400)
        return body
    except json.JSONDecodeError:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)


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
                emit({"type": "error", "message": str(exc)})
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
        wall_deadline = (
            time.monotonic()
            + float(settings.agent_max_runtime_seconds)
            + 120.0
        )

        async def event_stream():
            try:
                yield "retry: 3000\n\n"
                eid = 0
                client_gone = False
                while True:
                    if await request.is_disconnected():
                        if not client_gone:
                            client_gone = True
                            cancel_event.set()
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
                        "Request cancelled"
                        if outcome.cancelled
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
                    send_terminal_error = outcome.db_post_id is not None or (
                        not outcome.worker_emitted_error
                    )
                    if send_terminal_error:
                        yield sse_format_event(eid, err_ev)
                        eid += 1
            finally:
                await asyncio.shield(ensure_persist())
                sem.release()
                logger.info("[%s] stream generator finished", trace_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception:
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
