"""Tunable parameters for the Phase 0 acoustic spike."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcousticFingerprintConfig:
    sample_rate: int = 16000
    window_ms: int = 200
    hop_ms: int = 100
    stft_frame_ms: int = 25
    stft_hop_ms: int = 10
    mel_bands: int = 32
    mfcc_count: int = 13
    feature_type: str = 'logmel_mfcc_v1'
    fingerprint_min_dbfs: float = -50.0
    buffer_ttl_ms: int = 5000
    max_lag_ms: int = 1200
    lag_step_ms: int = 100
    sequence_windows: int = 3
    seller_threshold: float = 0.72
    customer_threshold: float = 0.45
    margin_threshold: float = 0.08
    hysteresis_k: int = 3
    hysteresis_m: int = 4
    classification_delay_ms: int = 500
