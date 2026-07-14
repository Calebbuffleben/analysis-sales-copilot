"""Typed payloads for the acoustic fingerprint spike."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AcousticClass = Literal['seller', 'customer', 'unknown']


@dataclass(frozen=True)
class AudioFingerprint:
    version: int
    user_id: str
    seller_room_id: str
    meeting_id: str
    seq: int
    window_duration_ms: int
    capture_time_ms: int
    energy_dbfs: float
    feature_type: str
    features: tuple[float, ...]

    def as_vector(self) -> list[float]:
        return list(self.features)


@dataclass(frozen=True)
class CorrelationResult:
    acoustic_class: AcousticClass
    matched_seller_id: str | None
    confidence: float
    lag_ms: int
    best_score: float
    second_best_score: float


@dataclass(frozen=True)
class LabeledWindow:
    start_ms: int
    end_ms: int
    ground_truth: AcousticClass
    matched_seller_id: str | None = None


@dataclass
class CorpusSession:
    session_id: str
    scenario: str
    seller_user_id: str
    listener_user_id: str
    meeting_id: str
    seller_room_id: str
    mic_pcm: bytes
    loopback_pcm: bytes
    sample_rate: int = 16000
    channels: int = 1
    labels: list[LabeledWindow] = field(default_factory=list)
    simulated_lag_ms: int = 0
