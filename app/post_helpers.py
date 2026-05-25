"""Shared helpers for post API (serialization, request parsing, agent persist)."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import USER_CANCEL_MESSAGE
from app.crud import create_post, get_post
from app.models import Post
from instagram.agent.orchestrator import agent_result_to_persist_content
from instagram.agent.runner import AgentResult
from instagram.agent.usage_tracking import UsageLedger

USER_QUERY_MAX_LEN = 16_000
SESSION_SUMMARY_MAX_LEN = 8_000
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


@dataclass
class AgentRunOutcome:
    """Filled by the worker thread; DB fields filled by persist (same process)."""

    usage_ledger: UsageLedger
    result: AgentResult | None = None
    error: Exception | None = None
    cancelled: bool = False
    user_cancel_notified: bool = False
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
                USER_CANCEL_MESSAGE
                if outcome.user_cancel_notified
                else (
                    USER_CANCEL_MESSAGE
                    if outcome.cancelled
                    else _debug_error_text(err, debug)
                )
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
