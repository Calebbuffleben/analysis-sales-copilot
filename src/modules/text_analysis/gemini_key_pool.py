"""Gemini API key pool with tenant-sticky routing and per-key RPM limits."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable

from ...config.settings import Settings
from ...metrics.realtime_metrics import (
    GEMINI_KEY_CALLS_TOTAL,
    GEMINI_KEY_RPM_LIMITED_TOTAL,
    GEMINI_POOL_SLOTS,
)
from .gemini_analyzer import GeminiAnalyzer

logger = logging.getLogger(__name__)

AnalyzerFactory = Callable[[str, str, int], GeminiAnalyzer]


@dataclass
class GeminiKeySlot:
    """One Gemini API key plus its own sliding-window RPM limiter."""

    index: int
    analyzer: GeminiAnalyzer
    rpm_limit: int
    rpm_window_sec: float
    _call_timestamps: Deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_acquire(self, now: float | None = None) -> bool:
        """Reserve one call if this slot has RPM capacity."""
        current = time.time() if now is None else now
        with self._lock:
            self._prune_locked(current)
            if len(self._call_timestamps) >= self.rpm_limit:
                GEMINI_KEY_RPM_LIMITED_TOTAL.labels(slot=str(self.index)).inc()
                return False
            self._call_timestamps.append(current)
            GEMINI_KEY_CALLS_TOTAL.labels(slot=str(self.index)).inc()
            return True

    def can_call(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        with self._lock:
            self._prune_locked(current)
            return len(self._call_timestamps) < self.rpm_limit

    def next_available_ms(self, now: float | None = None) -> int:
        current = time.time() if now is None else now
        with self._lock:
            self._prune_locked(current)
            if len(self._call_timestamps) < self.rpm_limit:
                return int(current * 1000)
            return int((self._call_timestamps[0] + self.rpm_window_sec) * 1000)

    def _prune_locked(self, now: float) -> None:
        while self._call_timestamps and (
            now - self._call_timestamps[0]
        ) > self.rpm_window_sec:
            self._call_timestamps.popleft()


class GeminiKeyPool:
    """Routes tenants to Gemini API keys and enforces per-key quotas."""

    def __init__(self, slots: Iterable[GeminiKeySlot], routing: str = 'tenant') -> None:
        self._slots = tuple(slots)
        if not self._slots:
            raise ValueError('GeminiKeyPool requires at least one key slot')
        if routing != 'tenant':
            raise ValueError(f'Unsupported Gemini key routing: {routing}')
        self._routing = routing
        GEMINI_POOL_SLOTS.set(len(self._slots))
        logger.info(
            'Gemini key pool initialized | slots=%s | rpm_per_slot=%s | '
            'window_sec=%s | routing=%s',
            len(self._slots),
            self._slots[0].rpm_limit,
            self._slots[0].rpm_window_sec,
            self._routing,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        analyzer_factory: AnalyzerFactory | None = None,
    ) -> 'GeminiKeyPool':
        factory = analyzer_factory or (
            lambda key, model, index: GeminiAnalyzer(
                api_key=key,
                model_name=model,
                slot_index=index,
            )
        )  # noqa: E731 — small factory for default analyzer wiring
        keys = settings.effective_gemini_api_keys()
        if (
            settings.gemini_api_key
            and ',' in settings.gemini_api_key.strip()
            and not settings.gemini_api_keys
        ):
            logger.warning(
                'GEMINI_API_KEY contains commas but GEMINI_API_KEYS is unset; '
                'splitting GEMINI_API_KEY into %s pool slot(s). '
                'Prefer GEMINI_API_KEYS=key1,key2 for clarity.',
                len(keys),
            )
        slots = [
            GeminiKeySlot(
                index=index,
                analyzer=factory(key, settings.gemini_model, index),
                rpm_limit=settings.gemini_rpm_limit,
                rpm_window_sec=settings.gemini_rpm_window_sec,
            )
            for index, key in enumerate(keys)
        ]
        return cls(slots, routing=settings.gemini_key_routing)

    @classmethod
    def from_analyzer(
        cls,
        analyzer: GeminiAnalyzer,
        *,
        rpm_limit: int,
        rpm_window_sec: float,
    ) -> 'GeminiKeyPool':
        return cls(
            (
                GeminiKeySlot(
                    index=0,
                    analyzer=analyzer,
                    rpm_limit=rpm_limit,
                    rpm_window_sec=rpm_window_sec,
                ),
            ),
        )

    @property
    def slots(self) -> tuple[GeminiKeySlot, ...]:
        return self._slots

    def resolve_slot(self, tenant_id: str | None) -> GeminiKeySlot:
        return self.resolve_slots_ordered(tenant_id)[0]

    def resolve_slots_ordered(self, tenant_id: str | None) -> tuple[GeminiKeySlot, ...]:
        """Sticky primary slot, then remaining keys for auth failover."""
        routing_key = (tenant_id or 'default').strip() or 'default'
        digest = hashlib.sha256(routing_key.encode('utf-8')).hexdigest()
        primary_idx = int(digest[:8], 16) % len(self._slots)
        primary = self._slots[primary_idx]
        fallbacks = tuple(
            slot for slot in self._slots if slot is not primary
        )
        return (primary, *fallbacks)
