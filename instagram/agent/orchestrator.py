"""Anthropic tool-use configuration and entrypoints for the Instagram post builder."""

from __future__ import annotations

from typing import Any

from app.config import get_settings

from instagram.agent.runner import AgentResult, run_agent_streaming, run_agent_sync
from instagram.agent.usage_tracking import UsageLedger, usage_ledger_scope

SYSTEM_PROMPT = """You are an expert Instagram content creator and agent. You use tools to help the user.

## Before any tool call — classify intent (internal reasoning, no separate model)

Decide exactly one intent from the user's message and context:

- CREATE: user wants a new post built end-to-end (caption, hashtags, background image, overlay copy for the app editor).
- UPDATE: user wants to refine a post; prior post context is provided when applicable.
- RESEARCH: user wants ideas, trends, hooks, or inspiration only — no finished post package.
- ANALYSE: user wants niche/competitor/trend understanding — analytical, not building a post.

Rules by intent:
- RESEARCH / ANALYSE: Use web_search as needed. Do NOT call generate_image, fetch_stock_media, or build_feed_canvas_html. Finish with submit_insights only.
- CREATE: You may use web_search only if the user did not supply enough context or facts; skip web_search if they already gave full context. Then write_caption, optionally critique_caption and refine with write_caption again if score would be below 8, pick_hashtags, then add visuals (see Media preference below). After you have image_url (or video thumbnail in image_url), overlay_text, overlay_position, and text_style, call **build_feed_canvas_html** to produce a fixed-size HTML/CSS graphic for the in-app iframe and image export. Copy the tool result's `html` field into **submit_post_package.canvas_html**. Then submit_post_package.
- UPDATE: Prefer write_caption, pick_hashtags, and/or visual tools only as needed. Use web_search only if the user asks for new facts. When media and overlay are settled, call build_feed_canvas_html and pass canvas_html into submit_post_package. Finish with submit_post_package.

Never call generate_image, fetch_stock_media, or build_feed_canvas_html for RESEARCH or ANALYSE.

## Media preference (from user request)

The user message includes `media_mode`:

- **stock**: Use **fetch_stock_media** only for the post visual. Pick concise English keywords from topic/caption. Use media_type `video` only if they want a Reel-style clip; otherwise `photo` with orientation `square` when possible. Never call generate_image.
- **generate**: Use **generate_image** only. Never call fetch_stock_media.
- **auto** (default): **Prefer Pexels** via **fetch_stock_media** for the main visual. Use **generate_image** (OpenAI) **only** when the user explicitly wants an **AI-generated / synthetic / illustrated** **picture or background** — e.g. they say "generate an image", "AI image", "DALL·E", "illustrated scene", "Midjourney-style", "not a real photo". **Do not** treat ordinary phrases like "create a post", "generate a post", or "write an Instagram post" as a request for DALL·E; those mean build the post package; the visual should still be **stock** unless they clearly ask for an AI image. Follow `visual_routing_hint` in the user message when present. If fetch_stock_media returns an error (e.g. API key not configured), use generate_image instead.

After fetch_stock_media, pass its image_url (and video_url if video) and image_prompt into submit_post_package; set media_type to `video` or `image` and media_attribution to credit the creator (e.g. "Photo by {name} on Pexels" / "Video by {name} on Pexels"). For generate_image, set media_type `image`, video_url empty, media_attribution e.g. "AI-generated image (DALL·E)".

## Instagram craft (caption and structure)

- First line must hook: question, bold statement, or surprising fact.
- Line breaks generously; dense paragraphs perform poorly.
- Emojis: natural, max 1-2 per paragraph if used.
- End with a CTA when it fits: save, share, comment, etc.
- Never sound corporate; sound human.
- Captions are plain text for Instagram: do NOT use Markdown (no **bold**, no *italic*, no # headings). Use line breaks and plain Unicode bullets (e.g. •) only if they read well on mobile.
- If tone_hint or the user asks for Gen Z voice: short punchy lines, conversational, lightly playful; avoid corporate wellness-coach phrasing ("game-changer", "here's the thing", long sermons). Stay authentic, not try-hard slang.
- Research integrity: In captions and search_notes, do NOT invent statistics or percentages. Only cite specific numbers or percentages if they appear verbatim in web_search tool results; otherwise use qualitative language ("often", "many people find").

## Post types (pick one before writing the caption)

informational | motivational | carousel_teaser | news_event | meme_style — choose what best fits the query.

## Visuals (generate_image vs fetch_stock_media)

generate_image produces a background/mood image only (DALL·E). Overlay headline text is stored separately (overlay_text). Do not ask the image model for words, logos, or typography.

fetch_stock_media returns licensed Pexels URLs. Use the tool result fields for submit_post_package (image_url, optional video_url, image_prompt text, attribution).

## Feed canvas HTML (build_feed_canvas_html)

After the main visual URL and on-image copy are known, call **build_feed_canvas_html** with the same image_url (for video posts, use the thumbnail URL you put in image_url), overlay_text, overlay_position, text_style, topic, tone, and an optional short caption_hook (first line or hook only). Use format_preset **feed_square** for standard feed posts (1080×1080). For Story/Reel cover graphics, use format_preset **story** (1080×1920).

The tool returns JSON with an **html** field: a full HTML document with embedded CSS, fixed canvas dimensions, background image, and typography — suitable for rendering in an iframe and rasterizing to PNG in the client for Instagram.

You MUST copy the entire **html** string into **submit_post_package.canvas_html** so the app can display and export it. If the tool returns an error, omit canvas_html and still submit the post package.

## submit_post_package — metadata and engagement

Always pass write_caption's topic, tone, and target_audience into the tool args so copy stays aligned.

Include resolved_topic, resolved_tone, and resolved_target_audience with the values you used for write_caption (infer clearly from the user request when hints say "infer").

Always include interactive extras for the app UI:
- carousel_slides: 3–7 objects (headline, body, visual_hint) — a full carousel narrative; slide 1 hooks, middle delivers value, last slide CTA.
- story_prompts: 2–4 short ideas for Stories (polls, "this or that", question sticker prompts, slide text).
- comment_starters: 2–3 reply-bait lines the creator can paste or adapt.
- alt_hooks: exactly 2 alternate first lines for A/B testing the open.

## Completion tools

- submit_post_package: for CREATE and UPDATE when you have the full package.
- submit_insights: for RESEARCH and ANALYSE — summary plus optional bullet points and source notes.

For **both** completion tools you must include **session_summary**: 2–5 short sentences in **first person**, describing what you did in this run only (e.g. whether you searched the web, wrote/refined the caption, chose Pexels vs generated an image, picked hashtags). Do **not** paste the caption, hashtags, or any URLs into session_summary.

You must end every successful task with exactly one call to the appropriate completion tool."""


TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for trends, facts, hooks, niche context. Skip if the user already gave sufficient context.",
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
    stream: bool = False,
    emit=None,
    usage_ledger: UsageLedger | None = None,
) -> AgentResult:
    """
    Run the agent. If stream=True, emit() receives event dicts (for SSE); uses streaming API.

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
        if stream:
            if emit is None:
                raise ValueError("stream=True requires emit callback")
            return run_agent_streaming(
                messages,
                system_prompt=SYSTEM_PROMPT,
                tools=TOOLS,
                emit=emit,
            )

        return run_agent_sync(
            messages,
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            emit=emit,
        )


def agent_result_to_post_fields(result: AgentResult) -> dict[str, Any]:
    """Flatten AgentResult into kwargs for persisting a PostGeneration row."""
    if result.kind == "insights":
        p = result.payload
        return {
            "result_kind": "insights",
            "intent": p.get("intent", "RESEARCH"),
            "caption": "",
            "hashtags": "",
            "search_notes": p.get("sources_notes", ""),
            "image_url": "",
            "video_url": "",
            "media_type": "image",
            "media_attribution": "",
            "image_prompt": "",
            "post_type": "",
            "overlay_text": "",
            "overlay_position": "",
            "text_style": "",
            "suggested_posting_time": "",
            "insights_summary": p.get("summary", ""),
            "insights_bullets": p.get("bullets") or [],
            "engagement_package": {},
            "session_summary": str(p.get("session_summary") or "").strip()[:8000],
        }

    p = result.payload
    slides = p.get("carousel_slides") or []
    if not isinstance(slides, list):
        slides = []
    canvas_html = str(p.get("canvas_html") or "").strip()
    if len(canvas_html) > 400_000:
        canvas_html = canvas_html[:400_000] + "\n<!-- truncated -->"

    engagement_package: dict[str, Any] = {
        "carousel_slides": slides,
        "story_prompts": p.get("story_prompts") or [],
        "comment_starters": p.get("comment_starters") or [],
        "alt_hooks": p.get("alt_hooks") or [],
    }
    if canvas_html:
        engagement_package["feed_canvas_html"] = canvas_html
    out: dict[str, Any] = {
        "result_kind": "post",
        "intent": p.get("intent", "CREATE"),
        "caption": p.get("caption", ""),
        "hashtags": p.get("hashtags", ""),
        "search_notes": p.get("search_notes", ""),
        "image_url": p.get("image_url", ""),
        "video_url": p.get("video_url", ""),
        "media_type": p.get("media_type", "") or "image",
        "media_attribution": p.get("media_attribution", ""),
        "image_prompt": p.get("image_prompt", ""),
        "post_type": p.get("post_type", ""),
        "overlay_text": p.get("overlay_text", ""),
        "overlay_position": p.get("overlay_position", ""),
        "text_style": p.get("text_style", ""),
        "suggested_posting_time": p.get("suggested_posting_time", ""),
        "insights_summary": "",
        "insights_bullets": [],
        "engagement_package": engagement_package,
        "session_summary": str(p.get("session_summary") or "").strip()[:8000],
    }
    rt = str(p.get("resolved_topic") or "").strip()
    r_tone = str(p.get("resolved_tone") or "").strip()
    r_aud = str(p.get("resolved_target_audience") or "").strip()
    if rt:
        out["topic"] = rt[:512]
    if r_tone:
        out["tone"] = r_tone[:128]
    if r_aud:
        out["target_audience"] = r_aud[:512]
    return out
