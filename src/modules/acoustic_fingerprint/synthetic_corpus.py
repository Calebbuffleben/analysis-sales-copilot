"""Synthetic corpus generation for Phase 0 CI and offline experiments."""

from __future__ import annotations

import uuid

import numpy as np

from .config import AcousticFingerprintConfig
from .pcm_io import float32_to_pcm16
from .types import CorpusSession, LabeledWindow


def _speech_like_signal(
    duration_s: float,
    *,
    sample_rate: int,
    base_freq: float,
) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    envelope = 0.4 + 0.6 * np.sin(2 * np.pi * 2.5 * t)
    harmonics = (
        0.6 * np.sin(2 * np.pi * base_freq * t)
        + 0.25 * np.sin(2 * np.pi * (base_freq * 2.03) * t)
        + 0.15 * np.sin(2 * np.pi * (base_freq * 3.07) * t)
    )
    signal = envelope * harmonics
    return signal.astype(np.float32)


def _degrade_meet_loopback(signal: np.ndarray, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    degraded = signal.copy()
    kernel = np.ones(5, dtype=np.float32) / 5.0
    degraded = np.convolve(degraded, kernel, mode='same')
    degraded *= 0.85
    noise = rng.normal(0.0, 0.01, size=degraded.shape).astype(np.float32)
    return (degraded + noise).astype(np.float32)


def _apply_lag(signal: np.ndarray, *, lag_ms: int, sample_rate: int) -> np.ndarray:
    lag_samples = int(sample_rate * lag_ms / 1000)
    if lag_samples <= 0:
        return signal
    return np.concatenate(
        [np.zeros(lag_samples, dtype=np.float32), signal[:-lag_samples]],
    )


def build_self_roundtrip_session(
    *,
    seller_user_id: str = 'seller-a',
    listener_user_id: str = 'listener-b',
    lag_ms: int = 250,
) -> CorpusSession:
    sample_rate = AcousticFingerprintConfig().sample_rate
    mic = _speech_like_signal(3.0, sample_rate=sample_rate, base_freq=140.0)
    loopback = _degrade_meet_loopback(mic, seed=7)
    loopback = _apply_lag(loopback, lag_ms=lag_ms, sample_rate=sample_rate)
    return CorpusSession(
        session_id=f'synthetic-self-{uuid.uuid4().hex[:8]}',
        scenario='self_roundtrip',
        seller_user_id=seller_user_id,
        listener_user_id=listener_user_id,
        meeting_id='meet-synthetic',
        seller_room_id='room-synthetic',
        mic_pcm=float32_to_pcm16(mic),
        loopback_pcm=float32_to_pcm16(loopback),
        sample_rate=sample_rate,
        labels=[
            LabeledWindow(0, 3000, 'seller', seller_user_id),
        ],
        simulated_lag_ms=lag_ms,
    )


def build_customer_only_session(
    *,
    seller_user_id: str = 'seller-a',
    listener_user_id: str = 'listener-b',
) -> CorpusSession:
    sample_rate = AcousticFingerprintConfig().sample_rate
    customer = _speech_like_signal(2.5, sample_rate=sample_rate, base_freq=220.0)
    loopback = _degrade_meet_loopback(customer, seed=11)
    return CorpusSession(
        session_id=f'synthetic-customer-{uuid.uuid4().hex[:8]}',
        scenario='customer_only',
        seller_user_id=seller_user_id,
        listener_user_id=listener_user_id,
        meeting_id='meet-synthetic',
        seller_room_id='room-synthetic',
        mic_pcm=float32_to_pcm16(np.zeros_like(customer)),
        loopback_pcm=float32_to_pcm16(loopback),
        sample_rate=sample_rate,
        labels=[
            LabeledWindow(0, 2500, 'customer', None),
        ],
        simulated_lag_ms=0,
    )


def build_overlap_session(
    *,
    seller_user_id: str = 'seller-a',
    listener_user_id: str = 'listener-b',
) -> CorpusSession:
    sample_rate = AcousticFingerprintConfig().sample_rate
    seller = _speech_like_signal(2.0, sample_rate=sample_rate, base_freq=145.0)
    customer = _speech_like_signal(2.0, sample_rate=sample_rate, base_freq=230.0)
    overlap = seller + 0.8 * customer
    loopback = _degrade_meet_loopback(overlap, seed=19)
    return CorpusSession(
        session_id=f'synthetic-overlap-{uuid.uuid4().hex[:8]}',
        scenario='seller_customer_overlap',
        seller_user_id=seller_user_id,
        listener_user_id=listener_user_id,
        meeting_id='meet-synthetic',
        seller_room_id='room-synthetic',
        mic_pcm=float32_to_pcm16(seller),
        loopback_pcm=float32_to_pcm16(loopback),
        sample_rate=sample_rate,
        labels=[
            LabeledWindow(0, 2000, 'unknown', None),
        ],
        simulated_lag_ms=0,
    )


def default_synthetic_sessions() -> list[CorpusSession]:
    return [
        build_self_roundtrip_session(),
        build_customer_only_session(),
        build_overlap_session(),
    ]
