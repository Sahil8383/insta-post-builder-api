"""Post generation API (parity with former Django instagram/views.py)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crud import (
    create_post,
    create_post_usage,
    get_post,
    get_post_usage,
    list_posts,
    recent_posts_memory,
)
from app.database import get_db, get_session_factory
from app.models import PostGeneration
from instagram.agent.orchestrator import agent_result_to_post_fields, run_post_agent
from instagram.agent.runner import AgentResult, sse_format_event, sse_heartbeat
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


def _post_json_lean(pg: PostGeneration) -> dict[str, Any]:
    """Content-focused shape for API responses; omits empty optional strings/lists/objects."""
    is_insights = bool((pg.insights_summary or "").strip())
    out: dict[str, Any] = {
        "id": pg.id,
        "result_kind": "insights" if is_insights else "post",
        "status": pg.status,
        "created_at": pg.created_at.isoformat(),
    }

    def put_str(key: str, val: str | None, *, max_len: int | None = None) -> None:
        s = (val or "").strip()
        if not s:
            return
        if max_len is not None:
            s = s[:max_len]
        out[key] = s

    put_str("user_query", pg.user_query, max_len=10000)
    put_str("intent", pg.intent)
    put_str("topic", pg.topic)
    put_str("tone", pg.tone)
    put_str("target_audience", pg.target_audience)
    put_str("caption", pg.caption)
    put_str("hashtags", pg.hashtags)
    put_str("search_notes", pg.search_notes)
    put_str("post_type", pg.post_type)
    put_str("overlay_text", pg.overlay_text)
    put_str("overlay_position", pg.overlay_position)
    put_str("text_style", pg.text_style)
    put_str("suggested_posting_time", pg.suggested_posting_time)
    put_str("image_url", pg.image_url or "")
    put_str("video_url", pg.video_url or "")
    put_str("media_type", pg.media_type or "image")
    put_str("media_attribution", pg.media_attribution)
    put_str("image_prompt", pg.image_prompt)
    put_str("session_summary", pg.session_summary)
    put_str("error_message", pg.error_message)

    if is_insights:
        put_str("insights_summary", pg.insights_summary)
        bullets = pg.insights_bullets or []
        if bullets:
            out["insights_bullets"] = bullets
    else:
        pkg = pg.engagement_package or {}
        if isinstance(pkg, dict) and pkg:
            out["engagement_package"] = pkg

    if pg.parent_post_id is not None:
        out["parent_post_id"] = pg.parent_post_id

    return out


def _parent_context_block(pg: PostGeneration) -> str:
    parts = [
        f"id: {pg.id}",
        f"topic: {pg.topic}",
        f"tone: {pg.tone}",
        f"target_audience: {pg.target_audience}",
        f"post_type: {pg.post_type}",
        f"caption:\n{pg.caption}",
        f"overlay_text: {pg.overlay_text}",
        f"overlay_position: {pg.overlay_position}",
        f"text_style: {pg.text_style}",
        f"hashtags: {pg.hashtags}",
        f"image_url: {pg.image_url or ''}",
        f"video_url: {pg.video_url or ''}",
        f"media_type: {pg.media_type or 'image'}",
        f"media_attribution: {pg.media_attribution or ''}",
        f"image_prompt: {pg.image_prompt}",
    ]
    pkg = pg.engagement_package if isinstance(pg.engagement_package, dict) else {}
    if pkg.get("feed_canvas_html"):
        parts.append("prior_feed_canvas_html: (present; regenerate with build_feed_canvas_html if media or overlay changed)")
    return "\n".join(parts)


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
) -> tuple[PostGeneration | None, JSONResponse | None]:
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
    return (q[:512] if q else "General")


def _create_post_from_agent(
    session: Session,
    *,
    query: str,
    topic: str,
    tone: str,
    audience: str,
    parent_post: PostGeneration | None = None,
    parent_post_id: int | None = None,
    agent_payload: dict[str, Any],
) -> PostGeneration:
    row_topic = _row_topic_fallback(topic, query)
    url = agent_payload.get("image_url") or ""
    if url:
        agent_payload = {**agent_payload, "image_url": str(url)[:2048]}
    vurl = agent_payload.get("video_url") or ""
    if vurl:
        agent_payload = {**agent_payload, "video_url": str(vurl)[:2048]}
    base: dict[str, Any] = {
        "user_query": query[:10000] if query else "",
        "topic": row_topic,
        "tone": tone,
        "target_audience": audience,
        "status": PostGeneration.Status.COMPLETED,
        **agent_payload,
    }
    if not (str(base.get("tone") or "").strip()):
        base["tone"] = (tone or "")[:128]
    if not (str(base.get("target_audience") or "").strip()):
        base["target_audience"] = (audience or "")[:512]
    if parent_post is not None:
        base["parent_post"] = parent_post
    elif parent_post_id is not None:
        base["parent_post_id"] = parent_post_id
    return create_post(session, **base)


def _failed_row_kwargs(
    *,
    query: str,
    topic: str,
    tone: str,
    audience: str,
    parent_post: PostGeneration | None = None,
    parent_post_id: int | None = None,
    err_text: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "user_query": query[:10000] if query else "",
        "topic": _row_topic_fallback(topic, query),
        "tone": tone,
        "target_audience": audience,
        "status": PostGeneration.Status.FAILED,
        "error_message": err_text,
    }
    if parent_post is not None:
        row["parent_post"] = parent_post
    elif parent_post_id is not None:
        row["parent_post_id"] = parent_post_id
    return row


def _persist_agent_success(
    session: Session,
    *,
    query: str,
    topic: str,
    tone: str,
    audience: str,
    parent_post: PostGeneration | None = None,
    parent_post_id: int | None = None,
    agent_result: Any,
    usage_ledger: UsageLedger,
) -> PostGeneration:
    fields = agent_result_to_post_fields(agent_result)
    fields.pop("result_kind", None)
    pg = _create_post_from_agent(
        session,
        query=query,
        topic=topic,
        tone=tone,
        audience=audience,
        parent_post=parent_post,
        parent_post_id=parent_post_id,
        agent_payload=fields,
    )
    create_post_usage(session, pg.id, usage_ledger)
    return pg


def _sync_persist_stream_outcome(
    SessionLocal: Any,
    outcome: AgentRunOutcome,
    *,
    query: str,
    topic: str,
    tone: str,
    audience: str,
    parent_post_id: int | None,
    debug: bool,
) -> None:
    """Write success or failure row + usage; update outcome with DB ids."""
    session: Session = SessionLocal()
    try:
        err = outcome.error
        if err is not None:
            msg = (
                "Request cancelled"
                if outcome.cancelled
                else _debug_error_text(err, debug)
            )
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    query=query,
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    parent_post_id=parent_post_id,
                    err_text=msg,
                ),
            )
            create_post_usage(session, pg.id, outcome.usage_ledger)
            outcome.db_post_id = pg.id
            return

        if outcome.result is None:
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    query=query,
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    parent_post_id=parent_post_id,
                    err_text="No agent result",
                ),
            )
            create_post_usage(session, pg.id, outcome.usage_ledger)
            outcome.db_post_id = pg.id
            return

        try:
            pg = _persist_agent_success(
                session,
                query=query,
                topic=topic,
                tone=tone,
                audience=audience,
                parent_post_id=parent_post_id,
                agent_result=outcome.result,
                usage_ledger=outcome.usage_ledger,
            )
            rk = "insights" if (pg.insights_summary or "").strip() else "post"
            outcome.db_post_id = pg.id
            outcome.db_result_kind = rk
            outcome.db_session_summary = (pg.session_summary or "").strip()
        except Exception as exc:
            outcome.persist_error = exc
            pg = create_post(
                session,
                **_failed_row_kwargs(
                    query=query,
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    parent_post_id=parent_post_id,
                    err_text=_debug_error_text(exc, debug),
                ),
            )
            create_post_usage(session, pg.id, outcome.usage_ledger)
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
                        query=query,
                        topic=topic,
                        tone=tone,
                        audience=audience,
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
def post_list(limit: int = 50, db: Session = Depends(get_db)) -> JSONResponse:
    lim = min(limit, 200)
    rows = list_posts(db, lim)
    return JSONResponse({"results": [_post_json_lean(pg) for pg in rows]})


@router.get("/api/posts/{pk}/")
def post_detail(pk: int, db: Session = Depends(get_db)) -> JSONResponse:
    pg = get_post(db, pk)
    if pg is None:
        raise HTTPException(status_code=404, detail="Not found.")
    return JSONResponse(_post_json_lean(pg))


@router.get("/api/posts/{pk}/usage/")
def post_usage_detail(pk: int, db: Session = Depends(get_db)) -> JSONResponse:
    if get_post(db, pk) is None:
        raise HTTPException(status_code=404, detail="Not found.")
    u = get_post_usage(db, pk)
    if u is None:
        raise HTTPException(status_code=404, detail="No usage recorded for this post.")
    return JSONResponse(
        {
            "post_id": u.post_id,
            "usage_breakdown": u.usage_breakdown or {},
            "estimated_cost_usd": float(u.estimated_cost_usd),
        }
    )
