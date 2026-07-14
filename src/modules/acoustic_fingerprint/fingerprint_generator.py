"""Log-Mel + MFCC fingerprint extraction for Phase 0."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import AcousticFingerprintConfig
from .types import AudioFingerprint


def _import_numpy() -> type[np.ndarray]:
    return np.ndarray


def pcm16_to_float32(pcm: bytes, channels: int = 1) -> np.ndarray:
    if not pcm:
        return np.array([], dtype=np.float32)
    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    return (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def compute_energy_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    low_hz = 0.0
    high_hz = sample_rate / 2.0
    low_mel = _hz_to_mel(low_hz)
    high_mel = _hz_to_mel(high_hz)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i : i + 3]
        if center <= left or right <= center:
            continue
        for j in range(left, center):
            if 0 <= j < filters.shape[1]:
                filters[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < filters.shape[1]:
                filters[i, j] = (right - j) / max(right - center, 1)
    return filters


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels)
    k = np.arange(n_mfcc).reshape(-1, 1)
    return np.sqrt(2.0 / n_mels) * np.cos(math.pi * k * (2 * n + 1) / (2 * n_mels))


class FingerprintGenerator:
    """Extract compact normalized feature vectors from PCM windows."""

    def __init__(self, config: AcousticFingerprintConfig | None = None) -> None:
        self._config = config or AcousticFingerprintConfig()
        frame_len = max(1, int(self._config.sample_rate * self._config.stft_frame_ms / 1000))
        self._n_fft = 1
        while self._n_fft < frame_len:
            self._n_fft <<= 1
        self._frame_len = frame_len
        self._frame_hop = max(
            1,
            int(self._config.sample_rate * self._config.stft_hop_ms / 1000),
        )
        self._mel_filters = _build_mel_filterbank(
            self._config.sample_rate,
            self._n_fft,
            self._config.mel_bands,
        )
        self._dct = _dct_matrix(self._config.mfcc_count, self._config.mel_bands)

    @property
    def config(self) -> AcousticFingerprintConfig:
        return self._config

    def window_sample_count(self) -> int:
        return int(self._config.sample_rate * self._config.window_ms / 1000)

    def hop_sample_count(self) -> int:
        return int(self._config.sample_rate * self._config.hop_ms / 1000)

    def iter_windows(
        self,
        samples: np.ndarray,
        *,
        start_time_ms: int = 0,
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        win = self.window_sample_count()
        hop = self.hop_sample_count()
        if samples.size < win:
            return
        offset = 0
        while offset + win <= samples.size:
            start_ms = start_time_ms + int(offset * 1000 / self._config.sample_rate)
            end_ms = start_time_ms + int((offset + win) * 1000 / self._config.sample_rate)
            yield start_ms, end_ms, samples[offset : offset + win]
            offset += hop

    def extract_features(self, window: np.ndarray) -> tuple[float, ...]:
        if window.size == 0:
            return tuple()
        frames = []
        pos = 0
        while pos + self._frame_len <= window.size:
            frame = window[pos : pos + self._frame_len]
            windowed = frame * np.hanning(self._frame_len).astype(np.float32)
            spectrum = np.fft.rfft(windowed, n=self._n_fft)
            power = (np.abs(spectrum) ** 2).astype(np.float32)
            mel = self._mel_filters @ power
            mel = np.log(np.maximum(mel, 1e-10))
            frames.append(mel)
            pos += self._frame_hop
        if not frames:
            return tuple()
        mel_stack = np.stack(frames, axis=0)
        mel_mean = mel_stack.mean(axis=0)
        mel_std = mel_stack.std(axis=0)
        mfcc = self._dct @ mel_mean
        delta = np.diff(mfcc, prepend=mfcc[:1])
        vector = np.concatenate([mel_mean, mel_std, mfcc, delta]).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return tuple(float(x) for x in vector)

    def fingerprint_from_window(
        self,
        window: np.ndarray,
        *,
        user_id: str,
        seller_room_id: str,
        meeting_id: str,
        seq: int,
        capture_time_ms: int,
    ) -> AudioFingerprint | None:
        energy = compute_energy_dbfs(window)
        if energy < self._config.fingerprint_min_dbfs:
            return None
        features = self.extract_features(window)
        if not features:
            return None
        return AudioFingerprint(
            version=1,
            user_id=user_id,
            seller_room_id=seller_room_id,
            meeting_id=meeting_id,
            seq=seq,
            window_duration_ms=self._config.window_ms,
            capture_time_ms=capture_time_ms,
            energy_dbfs=energy,
            feature_type=self._config.feature_type,
            features=features,
        )

    def fingerprint_stream(
        self,
        pcm: bytes,
        *,
        user_id: str,
        seller_room_id: str,
        meeting_id: str,
        channels: int = 1,
        start_time_ms: int = 0,
    ) -> list[AudioFingerprint]:
        samples = pcm16_to_float32(pcm, channels)
        out: list[AudioFingerprint] = []
        seq = 0
        for start_ms, _end_ms, window in self.iter_windows(samples, start_time_ms=start_time_ms):
            fp = self.fingerprint_from_window(
                window,
                user_id=user_id,
                seller_room_id=seller_room_id,
                meeting_id=meeting_id,
                seq=seq,
                capture_time_ms=start_ms,
            )
            if fp is not None:
                out.append(fp)
                seq += 1
        return out
