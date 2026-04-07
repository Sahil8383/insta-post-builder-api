"""Per-request API usage and estimated cost tracking for the Instagram agent."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from decimal import Decimal
from typing import Any, Iterator

from app.config import get_settings

_ledger_var: ContextVar[UsageLedger | None] = ContextVar("instagram_usage_ledger", default=None)


class UsageLedger:
    """
    Accumulates billable usage for one user query / agent run.

    Thread-safe for mutations. Install into the current context with
    ``usage_ledger_scope`` from ``run_post_agent`` so tools and the runner
    record into the same ledger without threading parameters everywhere.
    """

    __slots__ = (
        "_lock",
        "_orch_in",
        "_orch_out",
        "_orch_req",
        "_tool_in",
        "_tool_out",
        "_tool_req",
        "_tavily",
        "_images",
        "_pexels",
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orch_in = 0
        self._orch_out = 0
        self._orch_req = 0
        self._tool_in = 0
        self._tool_out = 0
        self._tool_req = 0
        self._tavily = 0
        self._images: list[dict[str, str]] = []
        self._pexels = 0

    def add_anthropic_orchestrator(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._orch_in += max(0, int(input_tokens))
            self._orch_out += max(0, int(output_tokens))
            self._orch_req += 1

    def add_anthropic_tools(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._tool_in += max(0, int(input_tokens))
            self._tool_out += max(0, int(output_tokens))
            self._tool_req += 1

    def add_tavily_search(self, count: int = 1) -> None:
        with self._lock:
            self._tavily += max(0, int(count))

    def add_openai_image(self, *, model: str, size: str, quality: str) -> None:
        with self._lock:
            self._images.append(
                {
                    "model": model,
                    "size": size,
                    "quality": quality,
                }
            )

    def add_pexels_request(self, count: int = 1) -> None:
        with self._lock:
            self._pexels += max(0, int(count))

    def estimate_total_usd(self) -> Decimal:
        """Rough USD total from configured per-unit rates (see app.config.Settings)."""
        cfg = get_settings()
        inp_rate = Decimal(str(cfg.usage_anthropic_input_per_mtok_usd))
        out_rate = Decimal(str(cfg.usage_anthropic_output_per_mtok_usd))
        tavily_rate = Decimal(str(cfg.usage_tavily_per_search_usd))
        img_rate = Decimal(str(cfg.usage_openai_image_per_call_usd))
        pexels_rate = Decimal(str(cfg.usage_pexels_per_request_usd))

        with self._lock:
            orch_in, orch_out = self._orch_in, self._orch_out
            tool_in, tool_out = self._tool_in, self._tool_out
            tavily = self._tavily
            n_img = len(self._images)
            n_pexels = self._pexels

        tin = orch_in + tool_in
        tout = orch_out + tool_out
        anthropic_usd = (Decimal(tin) / Decimal(1_000_000)) * inp_rate + (
            Decimal(tout) / Decimal(1_000_000)
        ) * out_rate
        tavily_usd = Decimal(tavily) * tavily_rate
        openai_usd = Decimal(n_img) * img_rate
        pexels_usd = Decimal(n_pexels) * pexels_rate
        return (anthropic_usd + tavily_usd + openai_usd + pexels_usd).quantize(
            Decimal("0.000001")
        )

    def to_breakdown_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot for API responses and DB storage."""
        with self._lock:
            orch_in, orch_out, orch_req = self._orch_in, self._orch_out, self._orch_req
            tool_in, tool_out, tool_req = self._tool_in, self._tool_out, self._tool_req
            tavily = self._tavily
            images = list(self._images)
            pexels = self._pexels

        cfg = get_settings()
        total = self.estimate_total_usd()
        return {
            "anthropic": {
                "model": cfg.anthropic_model,
                "orchestrator": {
                    "input_tokens": orch_in,
                    "output_tokens": orch_out,
                    "requests": orch_req,
                },
                "caption_tools": {
                    "input_tokens": tool_in,
                    "output_tokens": tool_out,
                    "requests": tool_req,
                },
            },
            "openai": {
                "image_model": cfg.openai_image_model,
                "images": images,
            },
            "tavily": {"search_requests": tavily},
            "pexels": {"api_requests": pexels},
            "estimated_cost_usd": float(total),
            "pricing_note": "USD estimate from USAGE_* settings; verify against provider dashboards.",
        }


def get_usage_ledger() -> UsageLedger | None:
    return _ledger_var.get()


@contextmanager
def usage_ledger_scope(ledger: UsageLedger | None) -> Iterator[None]:
    if ledger is None:
        yield
        return
    token: Token = _ledger_var.set(ledger)
    try:
        yield
    finally:
        _ledger_var.reset(token)


def record_anthropic_usage(usage: Any, *, channel: str) -> None:
    """Record token usage from an Anthropic Message ``usage`` object."""
    ledger = get_usage_ledger()
    if ledger is None or usage is None:
        return
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    if channel == "orchestrator":
        ledger.add_anthropic_orchestrator(inp, out)
    else:
        ledger.add_anthropic_tools(inp, out)
