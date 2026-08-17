"""Lightweight turn-level prosody for Live feedback enrichment.

Prosody never invents coaching — it only produces acoustic snapshots that
enrich publish metadata and optional Live context nudges.
"""

from __future__ import annotations

import array
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

from .audio_diagnostics import compute_pcm_window_stats

# Same absolute sample threshold as audio_diagnostics / ManualVad speech gate.
_SPEECH_SAMPLE_THRESHOLD = 500

# ~20 ms frames at 16 kHz mono (320 samples × 2 bytes).
_FRAME_MS = 20
_BYTES_PER_SAMPLE = 2

EnergyLevel = Literal['low', 'mid', 'high']
HesitationHint = Literal['none', 'weak', 'moderate']

# Cap ~30 s of 16 kHz mono PCM so a stuck VAD cannot grow unbounded.
TURN_PCM_MAX_BYTES = 30 * 16_000 * _BYTES_PER_SAMPLE


@dataclass(frozen=True)
class ProsodySnapshot:
    """Acoustic snapshot for one client speech turn."""

    samples_count: int
    speech_count: int
    mean_rms_dbfs: Optional[float]
    speech_ratio: float
    duration_ms: int
    pause_count: int
    longest_pause_ms: int
    internal_pause_ratio: float
    energy_level: EnergyLevel
    hesitation_hint: HesitationHint
    energy_variance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def is_distinctive(self) -> bool:
        """True when worth surfacing in Live context nudge / UI."""
        if self.hesitation_hint != 'none':
            return True
        if self.energy_level == 'low' and self.duration_ms >= 800:
            return True
        if self.longest_pause_ms >= 600:
            return True
        return False

    def nudge_line(self) -> str:
        """One short factual line for activity_end nudge; empty if not distinctive."""
        if not self.is_distinctive():
            return ''
        parts: list[str] = []
        if self.energy_level == 'low':
            parts.append('energia baixa')
        elif self.energy_level == 'high':
            parts.append('energia alta')
        if self.longest_pause_ms >= 400:
            parts.append(f'pausa interna ~{self.longest_pause_ms}ms')
        elif self.hesitation_hint != 'none':
            parts.append(f'hesitação {self.hesitation_hint}')
        if not parts:
            return ''
        line = 'Prosódia turno anterior: ' + '; '.join(parts) + '.'
        return line[:180]


def analyze_turn_prosody(
    turn_pcm: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> ProsodySnapshot:
    """Compute window stats + internal pause heuristics for a turn PCM buffer."""
    ch = max(int(channels), 1)
    sr = max(int(sample_rate), 1)
    stats = compute_pcm_window_stats(
        turn_pcm,
        sample_rate=sr,
        channels=ch,
    )
    samples_count = int(stats['samples_count'] or 0)
    speech_count = int(stats['speech_count'] or 0)
    mean_rms = stats.get('mean_rms_dbfs')
    duration_s = float(stats.get('duration_seconds') or 0.0)
    duration_ms = int(round(duration_s * 1000.0))
    speech_ratio = (
        float(speech_count) / float(samples_count) if samples_count > 0 else 0.0
    )

    pause_count, longest_pause_ms, internal_pause_ratio, energy_variance = (
        _frame_pause_metrics(turn_pcm, sample_rate=sr, channels=ch)
    )
    energy_level = _energy_level(mean_rms if mean_rms is None else float(mean_rms))
    hesitation = _hesitation_hint(
        longest_pause_ms=longest_pause_ms,
        pause_count=pause_count,
        internal_pause_ratio=internal_pause_ratio,
        energy_level=energy_level,
        duration_ms=duration_ms,
    )

    return ProsodySnapshot(
        samples_count=samples_count,
        speech_count=speech_count,
        mean_rms_dbfs=None if mean_rms is None else float(mean_rms),
        speech_ratio=round(speech_ratio, 4),
        duration_ms=duration_ms,
        pause_count=pause_count,
        longest_pause_ms=longest_pause_ms,
        internal_pause_ratio=round(internal_pause_ratio, 4),
        energy_level=energy_level,
        hesitation_hint=hesitation,
        energy_variance=round(energy_variance, 4),
    )


def _energy_level(mean_rms_dbfs: Optional[float]) -> EnergyLevel:
    if mean_rms_dbfs is None:
        return 'mid'
    if mean_rms_dbfs < -35.0:
        return 'low'
    if mean_rms_dbfs > -20.0:
        return 'high'
    return 'mid'


def _hesitation_hint(
    *,
    longest_pause_ms: int,
    pause_count: int,
    internal_pause_ratio: float,
    energy_level: EnergyLevel,
    duration_ms: int,
) -> HesitationHint:
    if duration_ms < 400:
        return 'none'
    if longest_pause_ms >= 800 or (
        pause_count >= 2 and internal_pause_ratio >= 0.25
    ):
        return 'moderate'
    if longest_pause_ms >= 400 or (
        pause_count >= 1 and internal_pause_ratio >= 0.15
    ):
        if energy_level == 'low' and longest_pause_ms >= 500:
            return 'moderate'
        return 'weak'
    return 'none'


def _frame_pause_metrics(
    turn_pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> tuple[int, int, float, float]:
    """Return pause_count, longest_pause_ms, internal_pause_ratio, energy_variance."""
    ch = max(channels, 1)
    frame_samples = max(1, int(sample_rate * _FRAME_MS / 1000.0))
    frame_bytes = frame_samples * ch * _BYTES_PER_SAMPLE
    if len(turn_pcm) < frame_bytes * 2:
        return 0, 0, 0.0, 0.0

    usable = len(turn_pcm) - (len(turn_pcm) % frame_bytes)
    pcm = array.array('h')
    pcm.frombytes(turn_pcm[:usable])

    frame_energies: list[float] = []
    speech_flags: list[bool] = []
    for i in range(0, len(pcm), frame_samples * ch):
        frame = pcm[i : i + frame_samples * ch]
        if ch > 1:
            mono = frame[::ch]
        else:
            mono = frame
        if not mono:
            continue
        # Frame "speech" if enough samples clear the same threshold as diagnostics.
        speech_n = sum(1 for s in mono if abs(s) >= _SPEECH_SAMPLE_THRESHOLD)
        is_speech = speech_n >= max(1, len(mono) // 8)
        speech_flags.append(is_speech)
        rms = math.sqrt(sum(s * s for s in mono) / len(mono))
        frame_energies.append(rms)

    if not speech_flags:
        return 0, 0, 0.0, 0.0

    # Trim leading/trailing silence so only internal gaps count as pauses.
    first = next((i for i, sp in enumerate(speech_flags) if sp), None)
    last = next((i for i, sp in enumerate(reversed(speech_flags)) if sp), None)
    if first is None or last is None:
        return 0, 0, 0.0, _variance(frame_energies)
    last_idx = len(speech_flags) - 1 - last
    if last_idx <= first:
        return 0, 0, 0.0, _variance(frame_energies)

    pause_count = 0
    longest_frames = 0
    silence_run = 0
    silence_total = 0
    span = last_idx - first + 1
    for flag in speech_flags[first : last_idx + 1]:
        if flag:
            if silence_run > 0:
                pause_count += 1
                longest_frames = max(longest_frames, silence_run)
                silence_total += silence_run
                silence_run = 0
        else:
            silence_run += 1
    if silence_run > 0:
        pause_count += 1
        longest_frames = max(longest_frames, silence_run)
        silence_total += silence_run

    longest_pause_ms = longest_frames * _FRAME_MS
    internal_pause_ratio = float(silence_total) / float(span) if span > 0 else 0.0
    return pause_count, longest_pause_ms, internal_pause_ratio, _variance(frame_energies)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
