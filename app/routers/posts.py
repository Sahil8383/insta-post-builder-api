"""Post generation API (parity with former Django instagram/views.py)."""

from __future__ import annotations

import json
import queue
import threading
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
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
from instagram.agent.runner import sse_format_event, sse_heartbeat
from instagram.agent.usage_tracking import UsageLedger

router = APIRouter(tags=["posts"])

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


def _normalize_media_mode(body: dict[str, Any]) -> str:
    m = str(body.get("media_mode") or "auto").strip().lower()
    if m in ("stock", "generate", "auto"):
        return m
    return "auto"


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


async def _read_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    raw = await request.body()
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
    raw = await _read_json_body(request)
    if isinstance(raw, JSONResponse):
        return raw
    body = raw

    query, topic, tone, audience = _resolve_query_and_hints(body)
    if query is None:
        return _MISSING_QUERY

    media_mode = _normalize_media_mode(body)

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

    usage_ledger = UsageLedger()
    event_q: queue.Queue[tuple[str, Any]] = queue.Queue()
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, BaseException] = {}

    def emit(ev: dict[str, Any]) -> None:
        event_q.put(("event", ev))

    def worker() -> None:
        try:
            result_holder["result"] = run_post_agent(
                query=query,
                tone=tone,
                target_audience=audience,
                topic=topic,
                parent_context=parent_context,
                recent_posts_memory=memory,
                media_mode=media_mode,
                emit=emit,
                usage_ledger=usage_ledger,
            )
        except BaseException as exc:
            error_holder["exc"] = exc
        finally:
            event_q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    settings = get_settings()

    def event_stream():
        yield "retry: 3000\n\n"
        eid = 0
        while True:
            kind, payload = event_q.get()
            if kind == "done":
                break
            ev = payload
            if ev.get("type") == "heartbeat":
                yield sse_heartbeat()
            else:
                yield sse_format_event(eid, ev)
                eid += 1

        stream_db = SessionLocal()
        try:
            exc = error_holder.get("exc")
            if exc is not None:
                pg = create_post(
                    stream_db,
                    **_failed_row_kwargs(
                        query=query,
                        topic=topic,
                        tone=tone,
                        audience=audience,
                        parent_post_id=parent_post_id,
                        err_text=_debug_error_text(exc, settings.debug),
                    ),
                )
                create_post_usage(stream_db, pg.id, usage_ledger)
                yield sse_format_event(
                    eid,
                    {"type": "error", "message": str(exc)},
                )
                return

            agent_result = result_holder.get("result")
            if agent_result is None:
                yield sse_format_event(eid, {"type": "error", "message": "No agent result"})
                return

            try:
                pg = _persist_agent_success(
                    stream_db,
                    query=query,
                    topic=topic,
                    tone=tone,
                    audience=audience,
                    parent_post_id=parent_post_id,
                    agent_result=agent_result,
                    usage_ledger=usage_ledger,
                )
                rk = (
                    "insights"
                    if (pg.insights_summary or "").strip()
                    else "post"
                )
                yield sse_format_event(
                    eid,
                    {
                        "type": "done",
                        "post_id": pg.id,
                        "result_kind": rk,
                        "session_summary": (pg.session_summary or "").strip(),
                    },
                )
            except Exception as exc:
                pg = create_post(
                    stream_db,
                    **_failed_row_kwargs(
                        query=query,
                        topic=topic,
                        tone=tone,
                        audience=audience,
                        parent_post_id=parent_post_id,
                        err_text=_debug_error_text(exc, settings.debug),
                    ),
                )
                create_post_usage(stream_db, pg.id, usage_ledger)
                yield sse_format_event(eid, {"type": "error", "message": str(exc)})
        finally:
            stream_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
