"""Throttle host multimodal observe while Gemini Live owns client audio."""

from __future__ import annotations


def should_skip_live_host_observe(
    *,
    live_host_context_enabled: bool,
    interval_ms: int,
    last_observe_ms: int,
    now_ms: int,
) -> bool:
    if not live_host_context_enabled or interval_ms <= 0:
        return False
    return (now_ms - last_observe_ms) < interval_ms
