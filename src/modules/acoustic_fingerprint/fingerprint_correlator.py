"""Correlate loopback windows against remote seller fingerprints."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import AcousticFingerprintConfig
from .fingerprint_buffer import FingerprintBuffer
from .fingerprint_generator import FingerprintGenerator, compute_energy_dbfs
from .types import AcousticClass, AudioFingerprint, CorrelationResult


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class FingerprintCorrelator:
    """Multi-window lag search with simple hysteresis."""

    def __init__(
        self,
        config: AcousticFingerprintConfig | None = None,
        *,
        generator: FingerprintGenerator | None = None,
    ) -> None:
        self._config = config or AcousticFingerprintConfig()
        self._generator = generator or FingerprintGenerator(self._config)
        self._recent_labels: deque[AcousticClass] = deque(maxlen=self._config.hysteresis_m)

    @property
    def config(self) -> AcousticFingerprintConfig:
        return self._config

    def _sequence_score(
        self,
        loopback_fps: list[AudioFingerprint],
        remote_fps: list[AudioFingerprint],
        *,
        lag_ms: int,
    ) -> float:
        if not loopback_fps or not remote_fps:
            return 0.0
        scores: list[float] = []
        for loop_fp in loopback_fps:
            target_time = loop_fp.capture_time_ms - lag_ms
            best = 0.0
            for remote_fp in remote_fps:
                if abs(remote_fp.capture_time_ms - target_time) <= self._config.hop_ms:
                    best = max(best, cosine_similarity(loop_fp.features, remote_fp.features))
            scores.append(best)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _best_match_for_seller(
        self,
        loopback_fps: list[AudioFingerprint],
        remote_fps: list[AudioFingerprint],
    ) -> tuple[float, int]:
        best_score = 0.0
        best_lag = 0
        for lag in range(-self._config.max_lag_ms, self._config.max_lag_ms + 1, self._config.lag_step_ms):
            score = self._sequence_score(loopback_fps, remote_fps, lag_ms=lag)
            if score > best_score:
                best_score = score
                best_lag = lag
        return best_score, best_lag

    def _apply_hysteresis(self, candidate: AcousticClass) -> AcousticClass:
        self._recent_labels.append(candidate)
        if candidate != 'seller':
            return candidate
        seller_count = sum(1 for label in self._recent_labels if label == 'seller')
        if seller_count >= self._config.hysteresis_k:
            return 'seller'
        return 'unknown'

    def correlate_window(
        self,
        loopback_window: np.ndarray,
        *,
        capture_time_ms: int,
        buffer: FingerprintBuffer,
        loopback_seq: int,
        seller_room_id: str,
        meeting_id: str,
    ) -> CorrelationResult:
        loop_fp = self._generator.fingerprint_from_window(
            loopback_window,
            user_id='loopback',
            seller_room_id=seller_room_id,
            meeting_id=meeting_id,
            seq=loopback_seq,
            capture_time_ms=capture_time_ms,
        )
        energy = compute_energy_dbfs(loopback_window)
        if loop_fp is None or energy < self._config.fingerprint_min_dbfs:
            return CorrelationResult(
                acoustic_class='unknown',
                matched_seller_id=None,
                confidence=0.0,
                lag_ms=0,
                best_score=0.0,
                second_best_score=0.0,
            )

        loopback_fps = [loop_fp]
        seller_scores: list[tuple[str, float, int]] = []
        for seller_id in buffer.all_user_ids():
            remote = buffer.candidates(
                seller_id,
                center_ms=capture_time_ms,
                max_lag_ms=self._config.max_lag_ms,
            )
            if not remote:
                continue
            score, lag = self._best_match_for_seller(loopback_fps, remote)
            seller_scores.append((seller_id, score, lag))

        if not seller_scores:
            acoustic_class: AcousticClass = 'unknown'
            if loop_fp is not None:
                acoustic_class = 'customer'
            return CorrelationResult(
                acoustic_class=acoustic_class,
                matched_seller_id=None,
                confidence=0.0,
                lag_ms=0,
                best_score=0.0,
                second_best_score=0.0,
            )

        seller_scores.sort(key=lambda item: item[1], reverse=True)
        best_id, best_score, best_lag = seller_scores[0]
        second_best = seller_scores[1][1] if len(seller_scores) > 1 else 0.0
        margin = best_score - second_best

        candidate: AcousticClass = 'unknown'
        confidence = best_score
        matched_id: str | None = None

        if best_score >= self._config.seller_threshold and margin >= self._config.margin_threshold:
            candidate = 'seller'
            matched_id = best_id
        elif best_score <= self._config.customer_threshold:
            candidate = 'customer'
        elif loop_fp is not None:
            candidate = 'customer'

        final_class = self._apply_hysteresis(candidate)
        if final_class == 'seller':
            matched_id = best_id
        elif final_class != 'seller':
            matched_id = None

        return CorrelationResult(
            acoustic_class=final_class,
            matched_seller_id=matched_id,
            confidence=confidence,
            lag_ms=best_lag,
            best_score=best_score,
            second_best_score=second_best,
        )

    def correlate_stream(
        self,
        loopback_pcm: bytes,
        *,
        remote_fingerprints: list[AudioFingerprint],
        seller_room_id: str,
        meeting_id: str,
        channels: int = 1,
        start_time_ms: int = 0,
        simulated_lag_ms: int = 0,
    ) -> list[CorrelationResult]:
        from .fingerprint_generator import pcm16_to_float32

        buffer = FingerprintBuffer(self._config.buffer_ttl_ms)
        adjusted = [
            AudioFingerprint(
                version=fp.version,
                user_id=fp.user_id,
                seller_room_id=fp.seller_room_id,
                meeting_id=fp.meeting_id,
                seq=fp.seq,
                window_duration_ms=fp.window_duration_ms,
                capture_time_ms=fp.capture_time_ms + simulated_lag_ms,
                energy_dbfs=fp.energy_dbfs,
                feature_type=fp.feature_type,
                features=fp.features,
            )
            for fp in remote_fingerprints
        ]
        buffer.add_many(adjusted)

        samples = pcm16_to_float32(loopback_pcm, channels)
        results: list[CorrelationResult] = []
        seq = 0
        for start_ms, _end_ms, window in self._generator.iter_windows(
            samples,
            start_time_ms=start_time_ms,
        ):
            result = self.correlate_window(
                window,
                capture_time_ms=start_ms,
                buffer=buffer,
                loopback_seq=seq,
                seller_room_id=seller_room_id,
                meeting_id=meeting_id,
            )
            results.append(result)
            seq += 1
        return results
