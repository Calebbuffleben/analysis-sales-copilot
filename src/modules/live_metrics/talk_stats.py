"""Talk-to-listen accumulator per meeting.

Ceiling: energy-based speech_ratio from PCM windows / VAD spans, not diarization.
Upgrade path: per-speaker STT timestamps if we add a transcript stream.

Monologue = accumulated host *speech* time since the customer last spoke
(silence contributes ~0ms; any customer turn resets it).
Moving window = host ratio over the last RECENT_WINDOW_MS of samples, so the
UI can distinguish "call started talkative" from "talkative right now".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

MONOLOGUE_MS = 180_000  # 3 minutes
YELLOW_RATIO = 0.70
COOLDOWN_MS = 60_000
RECENT_WINDOW_MS = 120_000  # 2-minute moving window


@dataclass
class TalkStats:
    host_speech_ms: int = 0
    customer_speech_ms: int = 0
    host_monologue_ms: int = 0
    last_yellow_alert_ms: int = 0
    # (ts_ms, host_speech_ms, customer_speech_ms) samples for the moving window
    _samples: deque = field(default_factory=deque)

    @property
    def host_ratio(self) -> float:
        total = self.host_speech_ms + self.customer_speech_ms
        if total <= 0:
            return 0.0
        return self.host_speech_ms / total

    def _record(self, now_ms: int, host_ms: int, customer_ms: int) -> None:
        self._samples.append((now_ms, host_ms, customer_ms))
        self._prune(now_ms)

    def _prune(self, now_ms: int) -> None:
        while self._samples and now_ms - self._samples[0][0] > RECENT_WINDOW_MS:
            self._samples.popleft()

    def recent_host_ratio(self, now_ms: int | None = None) -> float:
        """Host ratio over the last RECENT_WINDOW_MS (0.0 when no samples)."""
        now_ms = now_ms or int(time.time() * 1000)
        self._prune(now_ms)
        host = sum(sample[1] for sample in self._samples)
        customer = sum(sample[2] for sample in self._samples)
        total = host + customer
        if total <= 0:
            return 0.0
        return host / total


@dataclass
class TalkStatsStore:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _by_meeting: dict[str, TalkStats] = field(default_factory=dict)

    def _key(self, tenant_id: str, meeting_id: str) -> str:
        return f'{tenant_id}:{meeting_id}'

    def get(self, tenant_id: str, meeting_id: str) -> TalkStats:
        with self._lock:
            return self._by_meeting.setdefault(self._key(tenant_id, meeting_id), TalkStats())

    def observe_host(
        self,
        tenant_id: str,
        meeting_id: str,
        *,
        duration_ms: int,
        speech_ratio: float,
        now_ms: int | None = None,
    ) -> TalkStats:
        now_ms = now_ms or int(time.time() * 1000)
        speech_ms = int(max(0, duration_ms) * max(0.0, min(1.0, speech_ratio)))
        with self._lock:
            stats = self._by_meeting.setdefault(self._key(tenant_id, meeting_id), TalkStats())
            stats.host_speech_ms += speech_ms
            # Continuous-speech accounting: silence windows add ~0ms, so the
            # counter only grows while the host is actually talking.
            stats.host_monologue_ms += speech_ms
            stats._record(now_ms, speech_ms, 0)
            return stats

    def observe_customer(
        self,
        tenant_id: str,
        meeting_id: str,
        *,
        duration_ms: int,
        now_ms: int | None = None,
    ) -> TalkStats:
        now_ms = now_ms or int(time.time() * 1000)
        speech_ms = max(0, int(duration_ms))
        with self._lock:
            stats = self._by_meeting.setdefault(self._key(tenant_id, meeting_id), TalkStats())
            stats.customer_speech_ms += speech_ms
            stats.host_monologue_ms = 0
            stats._record(now_ms, 0, speech_ms)
            return stats

    def pop_yellow_alert(self, tenant_id: str, meeting_id: str) -> str | None:
        """Return yellow alert message once per cooldown if monologue exceeded."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            stats = self._by_meeting.get(self._key(tenant_id, meeting_id))
            if stats is None:
                return None
            if stats.host_monologue_ms < MONOLOGUE_MS:
                return None
            if now_ms - stats.last_yellow_alert_ms < COOLDOWN_MS:
                return None
            stats.last_yellow_alert_ms = now_ms
            return 'Vendedor monopolizou a fala por mais de 3 minutos'
