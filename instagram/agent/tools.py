"""Tool implementations for the Instagram post builder agent."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
import httpx
from openai import OpenAI

from app.config import get_settings
from tavily import TavilyClient

from instagram.agent.usage_tracking import (
    get_usage_ledger,
    record_anthropic_usage,
)


def web_search(query: str) -> str:
    settings = get_settings()
    if not settings.tavily_api_key:
        return "Search unavailable: TAVILY_API_KEY is not configured."
    client = TavilyClient(api_key=settings.tavily_api_key)
    resp = client.search(query=query.strip(), max_results=6)
    ledger = get_usage_ledger()
    if ledger is not None:
        ledger.add_tavily_search(1)
    lines: list[str] = []
    for r in resp.get("results") or []:
        title = r.get("title") or ""
        content = (r.get("content") or "")[:500]
        url = r.get("url") or ""
        lines.append(f"- {title}\n  {content}\n  {url}")
    if not lines:
        return "No search results returned. Proceed using your general knowledge."
    return "Research notes:\n" + "\n".join(lines)


def _anthropic_text(prompt: str, max_tokens: int = 2048) -> str:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    record_anthropic_usage(getattr(msg, "usage", None), channel="tools")
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def write_caption(
    topic: str,
    tone: str,
    target_audience: str,
    research_notes: str,
) -> str:
    prompt = f"""You are an expert Instagram copywriter.

Write ONE polished Instagram caption for this post.

Constraints:
- Maximum 2,200 characters (Instagram hard limit).
- Strong hook in the FIRST LINE only, ideally under 90 characters (scannable on the feed).
- Generous line breaks; avoid dense paragraphs. Prefer short lines (often one sentence per line).
- For informational posts: start with an optional one-line TL;DR after the hook (plain text, e.g. "TL;DR: ...") then 3-5 scannable tips or bullets using plain Unicode (•) or short numbered lines — not Markdown.
- Match tone: {tone}
- Topic: {topic}
- Target audience: {target_audience}
- If tone is Gen Z or youth-skewing: conversational, punchy, lightly playful; no corporate wellness voice; no fake statistics — only cite numbers that appear in the research notes below, otherwise stay qualitative.
- Emojis: natural, not forced; at most 1-2 per paragraph if any.
- End with a clear CTA when appropriate (e.g. save, share, comment).
- Never sound corporate; sound like a real person.
- Plain text only for Instagram: NO Markdown (no **asterisks** for bold, no _italic_, no # headings).

Research / trends to consider (may be empty). Do not invent percentages or stats not stated here:
{research_notes}

Return ONLY the caption text, no quotes or preamble."""
    text = _anthropic_text(prompt)
    if len(text) > 2200:
        text = text[:2197] + "..."
    return text


def _strip_code_fences(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if len(lines) < 2:
        return s
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


# Instagram feed square export size (px). Client iframe can scale down for preview.
FEED_CANVAS_WIDTH_PX = 1080
FEED_CANVAS_HEIGHT_PX = 1080


def build_feed_canvas_html(
    image_url: str,
    overlay_text: str,
    overlay_position: str = "center",
    text_style: str = "bold",
    tone: str = "",
    topic: str = "",
    caption_hook: str = "",
    format_preset: str = "feed_square",
) -> str:
    """
    Generate one self-contained HTML document (embedded CSS) for an Instagram-ready graphic.
    Returns JSON including `html` for submit_post_package.canvas_html.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return json.dumps({"error": "ANTHROPIC_API_KEY is not configured."})

    url = (image_url or "").strip()
    if not url:
        return json.dumps({"error": "image_url is required."})

    preset = (format_preset or "feed_square").strip().lower()
    if preset in ("story", "reel", "story_9_16"):
        w, h = 1080, 1920
        preset_label = "Instagram Story / Reel cover (9:16)"
    else:
        w, h = FEED_CANVAS_WIDTH_PX, FEED_CANVAS_HEIGHT_PX
        preset_label = "Instagram feed square (1:1)"

    pos = (overlay_position or "center").strip().lower()
    if pos not in ("top", "center", "bottom"):
        pos = "center"
    style = (text_style or "bold").strip().lower()
    if style not in ("bold", "minimal"):
        style = "bold"

    hook = (caption_hook or "").strip()[:400]
    topic_s = (topic or "").strip()[:300]
    tone_s = (tone or "").strip()[:120]
    overlay = (overlay_text or "").strip()[:180]

    hook_line = hook if hook else "(omit teaser line if empty)"

    prompt = f"""You are a senior UI designer. Output ONE complete HTML5 document only (no markdown, no preamble).

Goal: a static graphic for {preset_label}, fixed canvas {w}px wide by {h}px tall, ready to rasterize in the browser for Instagram.

## Content
- Background: full-bleed using this URL as a CSS background image (cover, centered):
  {url}
- Primary on-image headline (short; line breaks allowed):
  {overlay}
- Headline vertical region: {pos} third of the canvas (flex or absolute positioning).
- Text style: {style} — bold = high-contrast, strong type; minimal = lighter weight, more whitespace.
- Topic (mood only): {topic_s}
- Tone: {tone_s}
- Optional one-line teaser under headline: {hook_line}

## Technical rules (strict)
- Self-contained: all CSS in one <style> in <head>. No JavaScript.
- Root: one outer element with id `feed-canvas` with width {w}px; height {h}px; overflow hidden; position relative; box-sizing border-box.
- Fonts: system-ui stack, or one https Google Fonts <link> if needed.
- Readability on busy images: gradient scrim, soft shadow, or semi-transparent panel.
- No external images except the background URL above.
- No device frames or browser chrome.

Output ONLY the HTML document."""

    raw = _anthropic_text(prompt, max_tokens=8192)
    html = _strip_code_fences(raw)
    lower = html.lower().strip()
    if not lower.startswith("<!doctype") and not lower.startswith("<html"):
        return json.dumps(
            {
                "error": "Model did not return a valid HTML document.",
                "raw_preview": raw[:500],
            }
        )

    out: dict[str, Any] = {
        "ok": True,
        "format_preset": preset,
        "width_px": w,
        "height_px": h,
        "html": html,
        "hint": "Copy the full `html` string into submit_post_package.canvas_html.",
    }
    return json.dumps(out)


def critique_caption(caption: str, topic: str, tone: str) -> str:
    """Return a 1-10 score and concise feedback for Instagram performance."""
    prompt = f"""You evaluate Instagram captions for engagement potential.

Topic: {topic}
Tone target: {tone}

Caption to evaluate:
{caption}

Respond in this exact format (plain text):
Score: <integer 1-10>
Feedback: <one short paragraph: what works, what to improve for hooks, line breaks, CTA, authenticity; flag any Markdown or fake stats>
If score is below 8, add:
Rewrite_hint: <one line the writer should follow to improve>"""
    return _anthropic_text(prompt, max_tokens=512)


def pick_hashtags(topic: str, caption: str) -> str:
    prompt = f"""You pick Instagram hashtags for reach and relevance.

Topic: {topic}

Caption:
{caption}

Return between 12 and 18 hashtags as a single line, space-separated, each starting with #.
Mix broad reach tags with niche tags that fit the audience (e.g. Gen Z–relevant where appropriate). Only output the hashtag line, nothing else."""
    return _anthropic_text(prompt, max_tokens=512)


def generate_image(
    caption: str,
    topic: str,
    tone: str,
    image_prompt_hint: str,
) -> str:
    """Generate a background-only image via DALL·E; returns JSON string with url and prompt used."""
    settings = get_settings()
    if not settings.openai_api_key:
        return json.dumps({"error": "OPENAI_API_KEY is not configured."})

    oai = OpenAI(api_key=settings.openai_api_key)
    composed = f"""Square 1:1 Instagram feed background image (no UI mockup frame).

Topic mood: {topic}. Overall tone: {tone}.
Visual direction: {image_prompt_hint}

Scene/mood reference only (do NOT depict these as text):
{caption[:600]}

STRICT: No readable letters, words, numbers, logos, brand marks, signage, watermarks, or typography of any kind.
Abstract or photographic background only; text and headlines will be added in a separate design layer by the app."""

    img = oai.images.generate(
        model=settings.openai_image_model,
        prompt=composed[:4000],
        size="1024x1024",
        n=1,
        quality="standard",
    )
    item = img.data[0]
    url = item.url
    revised = getattr(item, "revised_prompt", None)
    ledger = get_usage_ledger()
    if ledger is not None:
        ledger.add_openai_image(
            model=settings.openai_image_model,
            size="1024x1024",
            quality="standard",
        )
    out: dict[str, Any] = {"image_url": url, "revised_prompt": revised or composed}
    return json.dumps(out)


PEXELS_PHOTOS_SEARCH = "https://api.pexels.com/v1/search"
PEXELS_VIDEOS_SEARCH = "https://api.pexels.com/videos/search"


def _pexels_best_mp4_link(video_files: list[Any]) -> str:
    mp4s: list[dict[str, Any]] = []
    for f in video_files:
        if not isinstance(f, dict):
            continue
        link = str(f.get("link") or "").strip()
        if not link:
            continue
        ft = str(f.get("file_type") or "").lower()
        if "mp4" in ft or link.lower().endswith(".mp4"):
            mp4s.append(f)
    if not mp4s:
        for f in video_files:
            if isinstance(f, dict) and str(f.get("link") or "").strip():
                return str(f["link"]).strip()
        return ""
    mp4s.sort(key=lambda x: int(x.get("width") or 0), reverse=True)
    return str(mp4s[0]["link"]).strip()


def fetch_stock_media(
    query: str,
    media_type: str = "photo",
    orientation: str = "square",
) -> str:
    """Search Pexels for a stock photo or video; returns JSON for the agent."""
    settings = get_settings()
    api_key = (settings.pexels_api_key or "").strip()
    if not api_key:
        return json.dumps({"error": "PEXELS_API_KEY is not configured."})

    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "Empty search query."})

    mt = (media_type or "photo").strip().lower()
    orient = (orientation or "").strip().lower()
    if orient not in ("landscape", "portrait", "square"):
        orient = "square" if mt != "video" else "landscape"

    headers = {"Authorization": api_key}
    ledger = get_usage_ledger()

    try:
        with httpx.Client(timeout=30.0) as client:
            if mt == "video":
                resp = client.get(
                    PEXELS_VIDEOS_SEARCH,
                    headers=headers,
                    params={"query": q, "per_page": 8, "orientation": orient},
                )
            else:
                resp = client.get(
                    PEXELS_PHOTOS_SEARCH,
                    headers=headers,
                    params={"query": q, "per_page": 8, "orientation": orient},
                )
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Pexels request failed: {exc!s}"})

    if ledger is not None:
        ledger.add_pexels_request(1)

    if resp.status_code != 200:
        return json.dumps(
            {
                "error": f"Pexels API HTTP {resp.status_code}: {resp.text[:300]}",
            }
        )

    data = resp.json()
    if mt == "video":
        videos = data.get("videos")
        if not isinstance(videos, list) or not videos:
            return json.dumps({"error": "No videos found for this query."})
        v = videos[0]
        if not isinstance(v, dict):
            return json.dumps({"error": "Unexpected Pexels video payload."})
        user = v.get("user")
        name = ""
        user_url = ""
        if isinstance(user, dict):
            name = str(user.get("name") or "").strip()
            user_url = str(user.get("url") or "").strip()
        if not name:
            name = "Unknown"
        page = str(v.get("url") or "").strip()
        thumbs = v.get("video_pictures")
        thumb = ""
        if isinstance(thumbs, list) and thumbs and isinstance(thumbs[0], dict):
            thumb = str(thumbs[0].get("picture") or "").strip()
        if not thumb:
            thumb = str(v.get("image") or "").strip()
        raw_files = v.get("video_files")
        files = raw_files if isinstance(raw_files, list) else []
        vlink = _pexels_best_mp4_link(files)
        if not vlink:
            return json.dumps({"error": "No playable video file in Pexels result."})
        out: dict[str, Any] = {
            "ok": True,
            "media_type": "video",
            "image_url": thumb,
            "video_url": vlink,
            "video_thumbnail_url": thumb,
            "photographer": name,
            "photographer_url": user_url,
            "pexels_page_url": page,
            "source_name": "Pexels",
            "alt_text": "",
            "image_prompt": f'Stock video from Pexels for "{q}" (credit: {name}).',
        }
        return json.dumps(out)

    photos = data.get("photos")
    if not isinstance(photos, list) or not photos:
        return json.dumps({"error": "No photos found for this query."})
    photo = photos[0]
    if not isinstance(photo, dict):
        return json.dumps({"error": "Unexpected Pexels photo payload."})
    src = photo.get("src")
    src_d: dict[str, Any] = src if isinstance(src, dict) else {}
    img_url = str(
        src_d.get("large2x") or src_d.get("large") or src_d.get("original") or ""
    ).strip()
    if not img_url:
        return json.dumps({"error": "No image URL in Pexels result."})
    ph = str(photo.get("photographer") or "").strip()
    ph_url = str(photo.get("photographer_url") or "").strip()
    page = str(photo.get("url") or "").strip()
    alt = str(photo.get("alt") or "").strip()
    out = {
        "ok": True,
        "media_type": "photo",
        "image_url": img_url,
        "video_url": "",
        "video_thumbnail_url": "",
        "photographer": ph,
        "photographer_url": ph_url,
        "pexels_page_url": page,
        "source_name": "Pexels",
        "alt_text": alt,
        "image_prompt": f'Stock photo from Pexels for "{q}" (credit: {ph}).',
    }
    return json.dumps(out)


TOOL_DISPATCH: dict[str, Any] = {
    "web_search": web_search,
    "write_caption": write_caption,
    "critique_caption": critique_caption,
    "pick_hashtags": pick_hashtags,
    "generate_image": generate_image,
    "fetch_stock_media": fetch_stock_media,
    "build_feed_canvas_html": build_feed_canvas_html,
}


def run_tool(name: str, tool_input: dict[str, Any]) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**tool_input)
    except TypeError as e:
        return f"Tool argument error for {name}: {e}"
    except Exception as e:
        return f"Tool error for {name}: {e!s}"


def parse_submit_payload(text: str) -> dict[str, Any] | None:
    """Extract JSON from assistant text if the model forgot to use a completion tool."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return None


def hashtags_to_string(value: Any) -> str:
    """Normalize hashtags from tool input (list or string) to a single space-separated line."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for x in value:
            s = str(x).strip()
            if not s:
                continue
            if not s.startswith("#"):
                s = f"#{s.lstrip('#')}"
            parts.append(s)
        return " ".join(parts)
    return str(value).strip()
