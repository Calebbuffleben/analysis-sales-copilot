"""Aggregated acoustic correlation metrics (no raw fingerprint logging)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class AcousticCorrelationMetrics:
    windows_total: int = 0
    by_class: Counter[str] = field(default_factory=Counter)
    seller_matches: int = 0
    confidence_sum: float = 0.0
    lag_sum_ms: float = 0.0
    publishes: int = 0
    publishes_rejected: int = 0

    def observe_window(
        self,
        *,
        acoustic_class: str,
        confidence: float = 0.0,
        lag_ms: float = 0.0,
    ) -> None:
        self.windows_total += 1
        self.by_class[acoustic_class] += 1
        if acoustic_class == 'seller':
            self.seller_matches += 1
        self.confidence_sum += confidence
        self.lag_sum_ms += lag_ms

    def observe_publish(self, *, rejected: bool = False) -> None:
        if rejected:
            self.publishes_rejected += 1
        else:
            self.publishes += 1

    def snapshot(self) -> dict[str, float | int | dict[str, int]]:
        unknown_rate = (
            self.by_class.get('unknown', 0) / self.windows_total
            if self.windows_total
            else 0.0
        )
        return {
            'windows_total': self.windows_total,
            'by_class': dict(self.by_class),
            'seller_match_rate': (
                self.seller_matches / self.windows_total if self.windows_total else 0.0
            ),
            'acoustic_unknown_rate': unknown_rate,
            'avg_confidence': (
                self.confidence_sum / self.windows_total if self.windows_total else 0.0
            ),
            'avg_lag_ms': (
                self.lag_sum_ms / self.windows_total if self.windows_total else 0.0
            ),
            'publishes': self.publishes,
            'publishes_rejected': self.publishes_rejected,
        }


# Process-wide metrics for ops scraping (resettable in tests).
CORRELATION_METRICS = AcousticCorrelationMetrics()
