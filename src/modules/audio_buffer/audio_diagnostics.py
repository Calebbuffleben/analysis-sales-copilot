"""PCM window diagnostics shared by buffer, STT, and feedback paths."""

from __future__ import annotations

import array
import math
from typing import Any, Dict, Optional

# Same threshold as TranscriptionPipelineService._apply_audio_window_stats
_SPEECH_SAMPLE_THRESHOLD = 500


def compute_pcm_window_stats(
    window_pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> Dict[str, Any]:
    """Compute level and speech heuristics for a 16-bit PCM window.

    Args:
        window_pcm: Raw PCM bytes (16-bit little-endian).
        sample_rate: Sample rate in Hz.
        channels: Number of interleaved channels (1 or 2).

    Returns:
        Dict with samples_count, speech_count, mean_rms_dbfs, peak_abs,
        duration_seconds, bytes_len.
    """
    ch = max(int(channels), 1)
    bytes_len = len(window_pcm)
    if bytes_len < 2 * ch or sample_rate <= 0:
        return {
            'samples_count': 0,
            'speech_count': 0,
            'mean_rms_dbfs': None,
            'peak_abs': 0,
            'duration_seconds': 0.0,
            'bytes_len': bytes_len,
        }

    sample_count = bytes_len // (2 * ch)
    if sample_count <= 0:
        return {
            'samples_count': 0,
            'speech_count': 0,
            'mean_rms_dbfs': None,
            'peak_abs': 0,
            'duration_seconds': 0.0,
            'bytes_len': bytes_len,
        }

    pcm_array = array.array('h')
    pcm_array.frombytes(window_pcm[: sample_count * 2 * ch])
    if ch > 1:
        mono_samples = pcm_array[::ch]
    else:
        mono_samples = pcm_array

    peak_abs = max((abs(s) for s in mono_samples), default=0)
    speech_count = sum(1 for s in mono_samples if abs(s) >= _SPEECH_SAMPLE_THRESHOLD)

    if not mono_samples:
        mean_rms_dbfs: Optional[float] = None
    else:
        rms = math.sqrt(sum(s * s for s in mono_samples) / len(mono_samples))
        if rms <= 0:
            mean_rms_dbfs = -120.0
        else:
            mean_rms_dbfs = round(20 * math.log10(rms / 32768.0), 2)

    bytes_per_second = sample_rate * ch * 2
    duration_seconds = bytes_len / max(bytes_per_second, 1)

    return {
        'samples_count': len(mono_samples),
        'speech_count': speech_count,
        'mean_rms_dbfs': mean_rms_dbfs,
        'peak_abs': peak_abs,
        'duration_seconds': duration_seconds,
        'bytes_len': bytes_len,
    }
