"""Shared STT logic used by in-process and process-pool transcription paths."""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from ..audio_buffer.audio_diagnostics import compute_pcm_window_stats
from .types import TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionSttConfig:
    """Configuration for one STT runtime (in-process or worker process)."""

    vad_filter: bool
    low_energy_dbfs_threshold: float
    default_language: Optional[str]
    empty_diagnostic_no_vad: bool


def transcribe_pcm_window(
    model: Any,
    window_pcm: bytes,
    meta: dict,
    *,
    config: TranscriptionSttConfig,
    diagnostic_executor: Optional[ThreadPoolExecutor],
    model_lock: threading.Lock,
) -> TranscriptionResult:
    """
    Transcribe one PCM window using an already-loaded faster-whisper model.

    Caller must hold `model_lock` for the duration of this call (same pattern as
    the original TranscriptionService: one lock for the main STT path; the
    optional diagnostic job acquires the lock after the caller releases it).
    """
    sample_rate = int(meta.get('sample_rate', 0) or 0)
    channels = max(int(meta.get('channels', 1)), 1)

    if not window_pcm:
        logger.info(
            '📝 STT skip | reason=no_pcm | meetingId=%s | participantId=%s',
            meta.get('meeting_id'),
            meta.get('participant_id'),
        )
        return TranscriptionResult(
            text='',
            confidence=0.0,
            segment_count=0,
            vad_filter_used=config.vad_filter,
            empty_reason='no_pcm',
        )

    stats = compute_pcm_window_stats(
        window_pcm,
        sample_rate=sample_rate,
        channels=channels,
    )

    mean_rms = stats.get('mean_rms_dbfs')
    if mean_rms is not None and mean_rms < config.low_energy_dbfs_threshold:
        logger.info(
            '📝 STT skip | reason=low_energy_precheck | meetingId=%s | '
            'participantId=%s | mean_rms_dbfs=%s | threshold=%s',
            meta.get('meeting_id'),
            meta.get('participant_id'),
            mean_rms,
            config.low_energy_dbfs_threshold,
        )
        return TranscriptionResult(
            text='',
            confidence=0.0,
            language=None,
            segment_count=0,
            vad_filter_used=config.vad_filter,
            empty_reason='low_energy',
        )

    np = _import_numpy()
    audio = np.frombuffer(window_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    language = _normalize_language(meta.get('language') or config.default_language)
    fallback_language = _normalize_language(
        meta.get('fallback_language') or config.default_language,
    )

    segment_list, transcript_text, confidence, detected_language = _run_transcribe(
        model=model,
        audio=audio,
        language=language,
        vad_filter=config.vad_filter,
    )

    diagnostic_no_vad_chars = 0
    empty_reason: Optional[str] = None
    used_fallback_language: Optional[str] = None

    if not transcript_text:
        empty_reason = _classify_empty_transcript(
            stats=stats,
            segment_count=len(segment_list),
            vad_filter=config.vad_filter,
            low_energy_dbfs_threshold=config.low_energy_dbfs_threshold,
        )
        if (
            diagnostic_executor is not None
            and config.empty_diagnostic_no_vad
            and config.vad_filter
            and len(segment_list) == 0
        ):
            diagnostic_executor.submit(
                _run_diagnostic_no_vad_only,
                model,
                model_lock,
                bytes(window_pcm),
                dict(meta),
                config,
            )

        if (
            not transcript_text
            and fallback_language
            and fallback_language != language
        ):
            (
                fb_list,
                fb_text,
                fb_confidence,
                fb_detected_language,
            ) = _run_transcribe(
                model=model,
                audio=audio,
                language=fallback_language,
                vad_filter=config.vad_filter,
            )
            if fb_text:
                segment_list = fb_list
                transcript_text = fb_text
                confidence = fb_confidence
                detected_language = fb_detected_language or fallback_language
                empty_reason = None
                used_fallback_language = fallback_language
                logger.info(
                    '📝 STT recovered with language fallback | meetingId=%s | '
                    'participantId=%s | fallback_language=%s | chars=%s | segments=%s',
                    meta.get('meeting_id'),
                    meta.get('participant_id'),
                    fallback_language,
                    len(transcript_text),
                    len(segment_list),
                )
            else:
                logger.info(
                    '📝 STT fallback language also empty | meetingId=%s | '
                    'participantId=%s | fallback_language=%s',
                    meta.get('meeting_id'),
                    meta.get('participant_id'),
                    fallback_language,
                )

        if not transcript_text:
            logger.info(
                '📝 STT empty | reason=%s | meetingId=%s | participantId=%s | '
                'vad_filter=%s | segments=%s | duration_s=%.3f | pcm_bytes=%s | '
                'mean_rms_dbfs=%s | speech_ratio=%.4f | peak_abs=%s | '
                'diag_no_vad_chars=%s | fallback_language=%s',
                empty_reason,
                meta.get('meeting_id'),
                meta.get('participant_id'),
                config.vad_filter,
                len(segment_list),
                float(stats.get('duration_seconds') or 0.0),
                stats.get('bytes_len'),
                stats.get('mean_rms_dbfs'),
                _speech_ratio(stats),
                stats.get('peak_abs'),
                diagnostic_no_vad_chars,
                fallback_language,
            )

    if transcript_text:
        logger.info(
            '📝 Transcription completed | meetingId=%s | participantId=%s | '
            'chars=%s | confidence=%.3f | vad_filter=%s | segments=%s | '
            'duration_s=%.3f | mean_rms_dbfs=%s | detected_language=%s | '
            'used_fallback_language=%s',
            meta.get('meeting_id'),
            meta.get('participant_id'),
            len(transcript_text),
            confidence,
            config.vad_filter,
            len(segment_list),
            float(stats.get('duration_seconds') or 0.0),
            stats.get('mean_rms_dbfs'),
            detected_language,
            used_fallback_language,
        )

    return TranscriptionResult(
        text=transcript_text,
        confidence=confidence,
        language=detected_language,
        used_fallback_language=used_fallback_language,
        segment_count=len(segment_list),
        vad_filter_used=config.vad_filter,
        empty_reason=empty_reason,
        diagnostic_no_vad_chars=diagnostic_no_vad_chars,
    )


def _run_diagnostic_no_vad_only(
    model: Any,
    model_lock: threading.Lock,
    window_pcm: bytes,
    meta: dict,
    config: TranscriptionSttConfig,
) -> None:
    with model_lock:
        try:
            np = _import_numpy()
            audio = np.frombuffer(window_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            language = _normalize_language(meta.get('language') or config.default_language)
            _nv_list, nv_text, _c, _l = _run_transcribe(
                model=model,
                audio=audio,
                language=language,
                vad_filter=False,
            )
            if nv_text:
                logger.info(
                    '📝 STT diagnostic | no-VAD rerun produced text | '
                    'chars=%s | meetingId=%s | suggests=VAD_or_filtering',
                    len(nv_text),
                    meta.get('meeting_id'),
                )
            else:
                logger.info(
                    '📝 STT diagnostic | no-VAD rerun also empty | meetingId=%s',
                    meta.get('meeting_id'),
                )
        except Exception:
            logger.exception(
                '📝 STT diagnostic | no-VAD rerun failed | meetingId=%s',
                meta.get('meeting_id'),
            )


def _speech_ratio(stats: dict) -> float:
    samples = int(stats.get('samples_count') or 0)
    if samples <= 0:
        return 0.0
    return float(stats.get('speech_count') or 0) / float(samples)


def _classify_empty_transcript(
    *,
    stats: dict,
    segment_count: int,
    vad_filter: bool,
    low_energy_dbfs_threshold: float,
) -> str:
    duration = float(stats.get('duration_seconds') or 0.0)
    if duration < 0.2:
        return 'window_too_short'
    mean_rms = stats.get('mean_rms_dbfs')
    if mean_rms is not None and mean_rms < low_energy_dbfs_threshold:
        return 'low_energy'
    if _speech_ratio(stats) < 0.01:
        return 'mostly_silent'
    if vad_filter and segment_count == 0:
        return 'empty_segments_vad_on'
    if not vad_filter and segment_count == 0:
        return 'empty_segments_vad_off'
    return 'no_text_other'


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            'numpy is required for transcription audio preprocessing.',
        ) from error
    return np


def _normalize_language(language: Optional[object]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip().lower()
    return value or None


def _calculate_confidence(segments: list[Any]) -> float:
    if not segments:
        return 0.0

    probabilities = []
    for segment in segments:
        avg_logprob = getattr(segment, 'avg_logprob', None)
        if avg_logprob is None:
            continue
        probabilities.append(max(0.0, min(1.0, math.exp(avg_logprob))))

    if not probabilities:
        return 0.0

    return sum(probabilities) / len(probabilities)


def _run_transcribe(
    *,
    model: Any,
    audio: Any,
    language: Optional[str],
    vad_filter: bool,
) -> tuple[list[Any], str, float, Optional[str]]:
    segments, info = model.transcribe(
        audio=audio,
        language=language or None,
        beam_size=1,
        vad_filter=vad_filter,
    )
    segment_list = list(segments)
    transcript_text = ' '.join(
        segment.text.strip()
        for segment in segment_list
        if getattr(segment, 'text', '').strip()
    ).strip()
    confidence = _calculate_confidence(segment_list)
    detected_language = getattr(info, 'language', None)
    return segment_list, transcript_text, confidence, detected_language
