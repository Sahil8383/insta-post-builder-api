"""Anthropic tool-use configuration and entrypoints for the Instagram post builder."""

from __future__ import annotations

import html as html_module
import threading
from typing import Any, Callable

from app.config import get_settings

from instagram.agent.prompts import SYSTEM_PROMPT
from instagram.agent.runner import AgentResult, run_agent_streaming
from instagram.agent.usage_tracking import UsageLedger, usage_ledger_scope

TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for trends, facts, hooks, niche context. Skip when the user already gave sufficient context or only needs copy/visuals without new facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_caption",
        "description": "Write or rewrite the Instagram caption using research notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "tone": {"type": "string"},
                "target_audience": {"type": "string"},
                "research_notes": {"type": "string"},
            },
            "required": ["topic", "tone", "target_audience", "research_notes"],
        },
    },
    {
        "name": "critique_caption",
        "description": "Rate caption 1-10 for Instagram performance and give short feedback; use before rewriting if unsure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "topic": {"type": "string"},
                "tone": {"type": "string"},
            },
            "required": ["caption", "topic", "tone"],
        },
    },
    {
        "name": "pick_hashtags",
        "description": "Pick 12-18 relevant hashtags for the caption (broad + niche).",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "caption": {"type": "string"},
            },
            "required": ["topic", "caption"],
        },
    },
    {
        "name": "generate_image",
        "description": "Generate a square background image (no text) via DALL·E. Returns JSON with image_url.",
        "input_schema": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "topic": {"type": "string"},
                "tone": {"type": "string"},
                "image_prompt_hint": {"type": "string"},
            },
            "required": ["caption", "topic", "tone", "image_prompt_hint"],
        },
    },
    {
        "name": "fetch_stock_media",
        "description": "Search Pexels for a stock photo or video (realistic licensed media). Returns JSON with image_url, optional video_url, photographer, pexels_page_url, image_prompt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short English search keywords (e.g. morning coffee desk remote work)",
                },
                "media_type": {
                    "type": "string",
                    "enum": ["photo", "video"],
                    "description": "photo for feed still; video for Reel-style clip",
                },
                "orientation": {
                    "type": "string",
                    "enum": ["square", "landscape", "portrait"],
                    "description": "Prefer square for feed photos when media_type is photo",
                },
            },
            "required": ["query", "media_type", "orientation"],
        },
    },
    {
        "name": "build_feed_canvas_html",
        "description": "Generate a fixed-dimension HTML+CSS document (Instagram feed 1:1 or Story 9:16) using the post image and overlay text; for iframe preview and client-side export to PNG. Returns JSON with html, width_px, height_px.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Same feed image URL as submit_post_package (thumbnail for video)",
                },
                "overlay_text": {
                    "type": "string",
                    "description": "On-image headline (same as submit_post_package.overlay_text)",
                },
                "overlay_position": {
                    "type": "string",
                    "enum": ["top", "center", "bottom"],
                },
                "text_style": {
                    "type": "string",
                    "enum": ["bold", "minimal"],
                },
                "tone": {"type": "string"},
                "topic": {"type": "string"},
                "caption_hook": {
                    "type": "string",
                    "description": "Optional short first-line or hook for subheadline (not full caption)",
                },
                "format_preset": {
                    "type": "string",
                    "enum": ["feed_square", "story", "reel", "story_9_16"],
                    "description": "feed_square = 1080×1080; story/reel/story_9_16 = 1080×1920",
                },
            },
            "required": [
                "image_url",
                "overlay_text",
                "overlay_position",
                "text_style",
                "tone",
                "topic",
            ],
        },
    },
    {
        "name": "submit_post_package",
        "description": "Complete CREATE/UPDATE: structured package for the app (caption + overlay + image_url from generate_image or fetch_stock_media; optional video_url; optional canvas_html from build_feed_canvas_html).",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "CREATE or UPDATE",
                },
                "post_type": {
                    "type": "string",
                    "description": "informational | motivational | carousel_teaser | news_event | meme_style",
                },
                "caption": {"type": "string"},
                "overlay_text": {
                    "type": "string",
                    "description": "3-5 word punchy headline for on-image overlay (rendered in the app, not in SD)",
                },
                "overlay_position": {
                    "type": "string",
                    "enum": ["top", "center", "bottom"],
                },
                "text_style": {
                    "type": "string",
                    "enum": ["bold", "minimal"],
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hashtags with or without leading #",
                },
                "search_notes": {
                    "type": "string",
                    "description": "Summary of web_search or note if skipped",
                },
                "image_url": {
                    "type": "string",
                    "description": "Feed image URL; for video posts use the thumbnail/poster URL here",
                },
                "video_url": {
                    "type": "string",
                    "description": "MP4 URL for Reel-style posts; empty string if image-only",
                },
                "media_type": {
                    "type": "string",
                    "enum": ["image", "video"],
                    "description": "image for photo or AI still; video when video_url is set",
                },
                "media_attribution": {
                    "type": "string",
                    "description": "Required credit line e.g. Photo by Name on Pexels, or AI-generated (DALL·E)",
                },
                "image_prompt": {"type": "string"},
                "suggested_posting_time": {
                    "type": "string",
                    "description": "e.g. morning | afternoon | evening",
                },
                "resolved_topic": {
                    "type": "string",
                    "description": "Topic string passed to write_caption; infer from user if hints were empty",
                },
                "resolved_tone": {
                    "type": "string",
                    "description": "Tone passed to write_caption; infer from user if hints were empty",
                },
                "resolved_target_audience": {
                    "type": "string",
                    "description": "Audience passed to write_caption; infer from user if hints were empty",
                },
                "carousel_slides": {
                    "type": "array",
                    "description": "3-7 carousel cards: hook → value → CTA",
                    "items": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "body": {"type": "string"},
                            "visual_hint": {
                                "type": "string",
                                "description": "Short art direction for this slide (no text in image)",
                            },
                        },
                        "required": ["headline", "body", "visual_hint"],
                    },
                },
                "story_prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 Story ideas: polls, questions, this-or-that",
                },
                "comment_starters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-3 reply-driving prompts",
                },
                "alt_hooks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exactly 2 alternate opening lines",
                },
                "session_summary": {
                    "type": "string",
                    "description": "2-5 sentences, first person: what you did this session (no caption/hashtags/URLs)",
                },
                "canvas_html": {
                    "type": "string",
                    "description": "Full HTML document from build_feed_canvas_html result field `html`; omit if tool failed",
                },
            },
            "required": [
                "intent",
                "post_type",
                "caption",
                "overlay_text",
                "overlay_position",
                "text_style",
                "hashtags",
                "search_notes",
                "image_url",
                "video_url",
                "media_type",
                "media_attribution",
                "image_prompt",
                "suggested_posting_time",
                "resolved_topic",
                "resolved_tone",
                "resolved_target_audience",
                "carousel_slides",
                "story_prompts",
                "comment_starters",
                "alt_hooks",
                "session_summary",
            ],
        },
    },
    {
        "name": "submit_insights",
        "description": "Complete RESEARCH/ANALYSE: no image generation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "RESEARCH or ANALYSE",
                },
                "summary": {"type": "string"},
                "bullets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional concise bullet points",
                },
                "sources_notes": {
                    "type": "string",
                    "description": "What you learned from search or general knowledge",
                },
                "session_summary": {
                    "type": "string",
                    "description": "2-5 sentences, first person: what you did this session (no pasted research text)",
                },
            },
            "required": ["intent", "summary", "sources_notes", "session_summary"],
        },
    },
]


def _visual_routing_hint(query: str) -> str:
    """Bias auto mode: Pexels by default; DALL-E only on explicit AI-image wording."""
    q = (query or "").lower()

    ai_visual_markers = (
        "generate an image",
        "generate a image",
        "generate the image",
        "generate image",
        "image generation",
        "ai-generated",
        "ai generated",
        "ai image",
        "ai picture",
        "artificial image",
        "synthetic image",
        "dall-e",
        "dall·e",
        "dalle",
        "openai image",
        "midjourney",
        "custom illustration",
        "cgi ",
        "3d render",
        "not a real photo",
        "not a real photograph",
        "fantasy illustration",
        "imagined scene",
    )
    for phrase in ai_visual_markers:
        if phrase in q:
            return (
                "Use generate_image (OpenAI): user wording asks for an AI/synthetic/illustrated "
                "visual, not stock photography."
            )

    stock_markers = (
        "stock photo",
        "real photo",
        "pexels",
        "authentic photo",
        "documentary photo",
    )
    for phrase in stock_markers:
        if phrase in q:
            return "Use fetch_stock_media: user asked for realistic/stock photography."

    return (
        "Use fetch_stock_media (Pexels) for the main visual by default. "
        "Switch to generate_image only if the user clearly requested an AI-generated image "
        "(not merely creating/generating the post text/package)."
    )


def build_user_message(
    *,
    query: str,
    tone: str,
    target_audience: str,
    topic: str,
    parent_context: str | None,
    recent_posts_memory: str,
    media_mode: str = "auto",
) -> str:
    mm = (media_mode or "auto").strip().lower()
    if mm not in ("stock", "generate", "auto"):
        mm = "auto"
    parts = [
        "User request:\n" + query.strip(),
        "",
        "Hints (use if helpful; you may infer missing pieces):",
        f"- topic_hint: {topic or '(infer from request)'}",
        f"- tone_hint: {tone or '(infer from request)'}",
        f"- target_audience_hint: {target_audience or '(infer from request)'}",
        f"- media_mode: {mm}",
        f"- visual_routing_hint: {_visual_routing_hint(query)}",
        "",
        "Recent memory:",
        recent_posts_memory,
    ]
    if parent_context:
        parts.extend(["", "Post being updated (context):", parent_context])
    parts.append(
        "\nClassify intent, use only relevant tools, then call the correct completion tool."
    )
    return "\n".join(parts)


def run_post_agent(
    *,
    query: str,
    tone: str = "",
    target_audience: str = "",
    topic: str = "",
    parent_context: str | None = None,
    recent_posts_memory: str = "(no prior completed posts in database)",
    media_mode: str = "auto",
    emit: Callable[[dict[str, Any]], None],
    usage_ledger: UsageLedger | None = None,
    cancel_event: threading.Event | None = None,
) -> AgentResult:
    """
    Run the agent with the streaming Messages API. ``emit`` receives SSE-shaped event dicts.

    Pass ``usage_ledger`` to record token counts and estimated USD for this query (also activates
    context for nested tool API calls).
    """
    if not get_settings().anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": build_user_message(
                query=query,
                tone=tone,
                target_audience=target_audience,
                topic=topic,
                parent_context=parent_context,
                recent_posts_memory=recent_posts_memory,
                media_mode=media_mode,
            ),
        }
    ]

    with usage_ledger_scope(usage_ledger):
        return run_agent_streaming(
            messages,
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            emit=emit,
            cancel_event=cancel_event,
        )


def _insights_payload_to_html(p: dict[str, Any]) -> str:
    summary_raw = (p.get("summary") or "").strip()
    summary = html_module.escape(summary_raw).replace("\n", "<br/>")
    bullets = p.get("bullets") or []
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"/>',
        "<title>Insights</title>",
        "<style>body{font-family:system-ui,sans-serif;padding:2rem;max-width:720px;line-height:1.5}</style>",
        "</head><body>",
        "<h1>Insights</h1>",
        f"<p>{summary}</p>",
    ]
    if isinstance(bullets, list) and bullets:
        parts.append("<ul>")
        for b in bullets:
            parts.append(f"<li>{html_module.escape(str(b).strip())}</li>")
        parts.append("</ul>")
    src = (p.get("sources_notes") or "").strip()
    if src:
        esc = html_module.escape(src).replace("\n", "<br/>")
        parts.append(f"<h2>Sources / notes</h2><p>{esc}</p>")
    parts.append("</body></html>")
    return "".join(parts)


def _insights_title_hint(p: dict[str, Any]) -> str:
    s = (p.get("summary") or "").strip()
    if s:
        line = s.split("\n", 1)[0].strip()
        return line[:512] if line else "Insights"
    return "Insights"


def agent_result_to_persist_content(result: AgentResult) -> dict[str, Any]:
    """Map AgentResult to DB-facing content: html_content, name hint, result_kind, session_summary."""
    if result.kind == "insights":
        p = result.payload
        return {
            "result_kind": "insights",
            "html_content": _insights_payload_to_html(p),
            "suggested_post_name": _insights_title_hint(p),
            "session_summary": str(p.get("session_summary") or "").strip()[:8000],
        }
    p = result.payload
    canvas_html = str(p.get("canvas_html") or "").strip()
    if len(canvas_html) > 400_000:
        canvas_html = canvas_html[:400_000] + "\n<!-- truncated -->"
    rt = str(p.get("resolved_topic") or "").strip()
    overlay = str(p.get("overlay_text") or "").strip()
    if rt:
        name_hint = rt[:512]
    elif overlay:
        name_hint = overlay[:512]
    else:
        name_hint = "Post"
    return {
        "result_kind": "post",
        "html_content": canvas_html,
        "suggested_post_name": name_hint,
        "session_summary": str(p.get("session_summary") or "").strip()[:8000],
    }
