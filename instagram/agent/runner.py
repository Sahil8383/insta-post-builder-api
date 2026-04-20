"""Anthropic tool loop with streaming Messages API (SSE-friendly)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import anthropic
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.config import get_settings
from instagram.agent.tools import (
    format_stream_tool_result,
    parse_submit_payload,
    run_tool,
)
from instagram.agent.usage_tracking import record_anthropic_usage

# One iteration = one Messages API round-trip (may include multiple tool calls in one turn).
MAX_ITERATIONS = 10

COMPLETION_TOOL_NAMES = frozenset({"submit_post_package", "submit_insights"})


@dataclass
class AgentResult:
    """Outcome of a completed agent run."""

    kind: str  # "post" | "insights"
    payload: dict[str, Any]


def _normalize_post_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields used after completion (persist + SSE). Full tool args stay in logs only."""
    return {
        "canvas_html": str(raw.get("canvas_html") or "").strip(),
        "session_summary": str(raw.get("session_summary") or "").strip(),
        "resolved_topic": str(raw.get("resolved_topic") or "").strip(),
        "overlay_text": str(raw.get("overlay_text") or "").strip(),
    }


def _normalize_insights_payload(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = raw.get("bullets")
    if bullets is None:
        bullets_list: list[Any] = []
    elif isinstance(bullets, list):
        bullets_list = bullets
    else:
        bullets_list = [str(bullets)]
    return {
        "intent": (raw.get("intent") or "RESEARCH").strip(),
        "summary": raw.get("summary") or "",
        "bullets": bullets_list,
        "sources_notes": raw.get("sources_notes") or "",
        "session_summary": str(raw.get("session_summary") or "").strip(),
    }


def _try_recover_payload_from_text(text: str) -> AgentResult | None:
    parsed = parse_submit_payload(text)
    if not parsed:
        return None
    if "summary" in parsed and "caption" not in parsed:
        return AgentResult(kind="insights", payload=_normalize_insights_payload(parsed))
    if "caption" in parsed:
        return AgentResult(kind="post", payload=_normalize_post_payload(parsed))
    return None


StreamEventFn = Callable[[dict[str, Any]], None]

_STREAM_RETRY_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def _assistant_content_from_response(response: Any) -> list[dict[str, Any]]:
    assistant_content: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            assistant_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return assistant_content


def _run_tool_round(
    response: Any,
    messages: list[dict[str, Any]],
    *,
    emit: StreamEventFn | None,
    emit_tool_starts: bool = True,
) -> AgentResult | None:
    """Append assistant message, execute tools, append tool results. Returns completion if any."""
    assistant_content = _assistant_content_from_response(response)
    messages.append({"role": "assistant", "content": assistant_content})

    tool_result_blocks: list[dict[str, Any]] = []
    completion: AgentResult | None = None

    for block in response.content:
        if block.type != "tool_use":
            continue
        name = block.name
        tool_input = block.input if isinstance(block.input, dict) else {}
        args_str = json.dumps(tool_input, default=str)

        if emit and emit_tool_starts:
            emit(
                {
                    "type": "tool-call-start",
                    "toolCallId": block.id,
                    "toolName": name,
                }
            )

        if name == "submit_post_package":
            completion = AgentResult(
                kind="post", payload=_normalize_post_payload(tool_input)
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({"status": "ok", "completed": True}),
                }
            )
            if emit:
                emit(
                    {
                        "type": "tool-call-end",
                        "toolCallId": block.id,
                        "toolName": name,
                        "result": "completed",
                    }
                )
        elif name == "submit_insights":
            completion = AgentResult(
                kind="insights", payload=_normalize_insights_payload(tool_input)
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({"status": "ok", "completed": True}),
                }
            )
            if emit:
                emit(
                    {
                        "type": "tool-call-end",
                        "toolCallId": block.id,
                        "toolName": name,
                        "result": "completed",
                    }
                )
        else:
            result_text = run_tool(name, tool_input)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
            if emit:
                emit(
                    {
                        "type": "tool-call-end",
                        "toolCallId": block.id,
                        "toolName": name,
                        "arguments": args_str,
                        "result": format_stream_tool_result(name, result_text),
                    }
                )

    if tool_result_blocks:
        messages.append({"role": "user", "content": tool_result_blocks})
    return completion


def run_agent_streaming(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str,
    tools: list[dict[str, Any]],
    emit: StreamEventFn,
    cancel_event: threading.Event | None = None,
) -> AgentResult:
    """Run the tool-use loop with client.messages.stream until a completion tool or max iterations."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    emit({"type": "assistant-message-id", "messageId": str(uuid.uuid4())})

    max_retries = max(1, int(settings.anthropic_stream_max_retries))
    deadline = time.monotonic() + float(settings.agent_max_runtime_seconds)

    for iteration in range(MAX_ITERATIONS):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Agent cancelled by client")
        if time.monotonic() > deadline:
            raise TimeoutError("Agent exceeded max runtime")

        emit({"type": "iteration", "index": iteration})
        emit({"type": "heartbeat"})

        response = None
        for attempt in range(max_retries):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Agent cancelled by client")
            try:
                with client.messages.stream(
                    model=settings.anthropic_model,
                    max_tokens=8192,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                ) as stream:
                    stream_tool_id = ""
                    stream_tool_name = ""
                    for event in stream:
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("Agent cancelled by client")
                        et = getattr(event, "type", None)
                        if et == "content_block_start":
                            cb = getattr(event, "content_block", None)
                            if cb is not None and getattr(cb, "type", None) == "tool_use":
                                stream_tool_id = cb.id
                                stream_tool_name = cb.name
                                emit(
                                    {
                                        "type": "tool-call-start",
                                        "toolCallId": cb.id,
                                        "toolName": cb.name,
                                    }
                                )
                        elif et == "content_block_delta":
                            delta = event.delta
                            dt = getattr(delta, "type", None)
                            if dt == "text_delta":
                                emit({"type": "reasoning-delta", "delta": delta.text})
                            elif dt == "input_json_delta":
                                if stream_tool_name not in COMPLETION_TOOL_NAMES:
                                    emit(
                                        {
                                            "type": "tool-call-delta",
                                            "toolCallId": stream_tool_id,
                                            "toolName": stream_tool_name,
                                            "arguments": delta.partial_json,
                                        }
                                    )
                    response = stream.get_final_message()
                break
            except _STREAM_RETRY_EXCEPTIONS:
                if attempt >= max_retries - 1:
                    raise
        assert response is not None

        record_anthropic_usage(
            getattr(response, "usage", None), channel="orchestrator"
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    recovered = _try_recover_payload_from_text(block.text)
                    if recovered:
                        return recovered
            break

        completion = _run_tool_round(
            response, messages, emit=emit, emit_tool_starts=False
        )
        if completion is not None:
            return completion

    raise RuntimeError(
        "Agent did not call submit_post_package or submit_insights within iteration limit."
    )


def sse_format_event(event_id: int, payload: dict[str, Any]) -> str:
    line = json.dumps(payload, default=str)
    return f"id: {event_id}\ndata: {line}\n\n"


def sse_heartbeat() -> str:
    return ": heartbeat\n\n"
