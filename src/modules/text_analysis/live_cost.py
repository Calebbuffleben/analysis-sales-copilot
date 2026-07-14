"""Estimate Gemini Live cost from usage_metadata and official pricing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# Official Gemini 3.1 Flash Live Preview paid-tier rates (USD / 1M tokens).
# Audio ≈ $0.005/min at 25 tokens/sec; context re-billing can dominate.
PRICE_AUDIO_INPUT_PER_M = 3.00
PRICE_TEXT_INPUT_PER_M = 0.75
PRICE_TEXT_OUTPUT_PER_M = 4.50
PRICE_AUDIO_OUTPUT_PER_M = 12.00


def _tokens(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def estimate_cost_usd(usage: Mapping[str, Any] | None) -> float:
    """Estimate USD cost for one usage_metadata payload."""
    if not usage:
        return 0.0

    # Prefer modality-specific counts when present.
    prompt_details = usage.get('prompt_tokens_details') or usage.get(
        'promptTokensDetails',
    )
    candidate_details = usage.get('candidates_tokens_details') or usage.get(
        'candidatesTokensDetails',
    )

    audio_in = 0
    text_in = 0
    audio_out = 0
    text_out = 0

    if isinstance(prompt_details, list):
        for item in prompt_details:
            if not isinstance(item, Mapping):
                continue
            modality = str(item.get('modality') or '').upper()
            count = _tokens(item.get('token_count') or item.get('tokenCount'))
            if modality == 'AUDIO':
                audio_in += count
            else:
                text_in += count
    else:
        text_in = _tokens(
            usage.get('prompt_token_count') or usage.get('promptTokenCount'),
        )

    if isinstance(candidate_details, list):
        for item in candidate_details:
            if not isinstance(item, Mapping):
                continue
            modality = str(item.get('modality') or '').upper()
            count = _tokens(item.get('token_count') or item.get('tokenCount'))
            if modality == 'AUDIO':
                audio_out += count
            else:
                text_out += count
    else:
        text_out = _tokens(
            usage.get('candidates_token_count')
            or usage.get('candidatesTokenCount')
            or usage.get('response_token_count')
            or usage.get('responseTokenCount'),
        )

    return (
        audio_in * PRICE_AUDIO_INPUT_PER_M
        + text_in * PRICE_TEXT_INPUT_PER_M
        + audio_out * PRICE_AUDIO_OUTPUT_PER_M
        + text_out * PRICE_TEXT_OUTPUT_PER_M
    ) / 1_000_000.0


@dataclass
class MeetingCostTracker:
    """Accumulates estimated Live cost for one meeting."""

    meeting_id: str
    max_cost_usd: float
    alert_cost_usd: float = 1.0
    total_usd: float = 0.0
    alerted: bool = False
    limited: bool = False
    audio_output_bytes: int = 0
    _turns: int = 0
    _started_wall_ms: int = field(default=0)

    def add_usage(self, usage: Mapping[str, Any] | None) -> float:
        delta = estimate_cost_usd(usage)
        self.total_usd += delta
        self._turns += 1
        if self.total_usd >= self.max_cost_usd:
            self.limited = True
        return delta

    def add_unexpected_audio(self, nbytes: int) -> None:
        self.audio_output_bytes += max(0, int(nbytes))

    def should_alert(self) -> bool:
        if self.alerted:
            return False
        if self.total_usd >= self.alert_cost_usd:
            self.alerted = True
            return True
        return False

    def projected_usd_per_hour(self, elapsed_ms: int) -> Optional[float]:
        if elapsed_ms <= 0 or self.total_usd <= 0:
            return None
        hours = elapsed_ms / 3_600_000.0
        if hours <= 0:
            return None
        return self.total_usd / hours
