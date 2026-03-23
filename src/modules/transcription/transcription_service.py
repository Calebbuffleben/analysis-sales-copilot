"""TranscriptionService: low-level service that turns audio windows into text.

This module is responsible only for calling the STT provider. It does not decide
what happens to the transcript; that is the responsibility of
TranscriptionPipelineService.

STT provider: faster-whisper (>= 1.0.0), which depends on ctranslate2.
System dependencies expected to be available in the runtime image:
- build-essential        (to build ctranslate2)
- ffmpeg                 (audio processing binary)
- libffi-dev
- libssl-dev
- pkg-config
- libsndfile1            (for soundfile support)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from ..audio_buffer.audio_diagnostics import compute_pcm_window_stats
from .types import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Turns a PCM window into transcript text using faster-whisper."""

    def __init__(
        self,
        model_size: str = 'small',
        device: str = 'cpu',
        compute_type: str = 'int8',
        vad_filter: bool = True,
        empty_diagnostic_no_vad: bool = False,
        low_energy_dbfs_threshold: float = -50.0,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._vad_filter = vad_filter
        self._empty_diagnostic_no_vad = empty_diagnostic_no_vad
        self._low_energy_dbfs_threshold = low_energy_dbfs_threshold
        self._model: Optional[Any] = None

    def transcribe(self, window_pcm: bytes, meta: dict) -> TranscriptionResult:
        """Transcribe a single PCM audio window.

        Args:
            window_pcm: Raw PCM bytes for the current window.
            meta: Context (e.g. sample_rate, channels, meeting_id, participant_id).

        Returns:
            A `TranscriptionResult` with text and confidence.
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
                vad_filter_used=self._vad_filter,
                empty_reason='no_pcm',
            )

        stats = compute_pcm_window_stats(
            window_pcm,
            sample_rate=sample_rate,
            channels=channels,
        )

        np = self._import_numpy()
        audio = np.frombuffer(window_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        model = self._get_model()
        language = meta.get('language')

        segments, info = model.transcribe(
            audio=audio,
            language=language or None,
            beam_size=1,
            vad_filter=self._vad_filter,
        )

        segment_list = list(segments)
        transcript_text = ' '.join(
            segment.text.strip()
            for segment in segment_list
            if getattr(segment, 'text', '').strip()
        ).strip()

        confidence = self._calculate_confidence(segment_list)
        detected_language = getattr(info, 'language', None)

        diagnostic_no_vad_chars = 0
        empty_reason: Optional[str] = None

        if not transcript_text:
            empty_reason = self._classify_empty_transcript(
                stats=stats,
                segment_count=len(segment_list),
            )
            if (
                self._empty_diagnostic_no_vad
                and self._vad_filter
                and len(segment_list) == 0
            ):
                segments_nv, _info_nv = model.transcribe(
                    audio=audio,
                    language=language or None,
                    beam_size=1,
                    vad_filter=False,
                )
                nv_list = list(segments_nv)
                nv_text = ' '.join(
                    segment.text.strip()
                    for segment in nv_list
                    if getattr(segment, 'text', '').strip()
                ).strip()
                diagnostic_no_vad_chars = len(nv_text)
                if diagnostic_no_vad_chars > 0:
                    logger.info(
                        '📝 STT diagnostic | no-VAD rerun produced text | '
                        'chars=%s | meetingId=%s | suggests=VAD_or_filtering',
                        diagnostic_no_vad_chars,
                        meta.get('meeting_id'),
                    )
                    if empty_reason == 'empty_segments_vad_on':
                        empty_reason = 'vad_likely_suppressed_speech'
                else:
                    logger.info(
                        '📝 STT diagnostic | no-VAD rerun also empty | meetingId=%s',
                        meta.get('meeting_id'),
                    )

            logger.info(
                '📝 STT empty | reason=%s | meetingId=%s | participantId=%s | '
                'vad_filter=%s | segments=%s | duration_s=%.3f | pcm_bytes=%s | '
                'mean_rms_dbfs=%s | speech_ratio=%.4f | peak_abs=%s | diag_no_vad_chars=%s',
                empty_reason,
                meta.get('meeting_id'),
                meta.get('participant_id'),
                self._vad_filter,
                len(segment_list),
                float(stats.get('duration_seconds') or 0.0),
                stats.get('bytes_len'),
                stats.get('mean_rms_dbfs'),
                self._speech_ratio(stats),
                stats.get('peak_abs'),
                diagnostic_no_vad_chars,
            )
        else:
            logger.info(
                '📝 Transcription completed | meetingId=%s | participantId=%s | '
                'chars=%s | confidence=%.3f | vad_filter=%s | segments=%s | '
                'duration_s=%.3f | mean_rms_dbfs=%s',
                meta.get('meeting_id'),
                meta.get('participant_id'),
                len(transcript_text),
                confidence,
                self._vad_filter,
                len(segment_list),
                float(stats.get('duration_seconds') or 0.0),
                stats.get('mean_rms_dbfs'),
            )

        return TranscriptionResult(
            text=transcript_text,
            confidence=confidence,
            language=detected_language,
            segment_count=len(segment_list),
            vad_filter_used=self._vad_filter,
            empty_reason=empty_reason,
            diagnostic_no_vad_chars=diagnostic_no_vad_chars,
        )

    def _speech_ratio(self, stats: dict) -> float:
        samples = int(stats.get('samples_count') or 0)
        if samples <= 0:
            return 0.0
        return float(stats.get('speech_count') or 0) / float(samples)

    def _classify_empty_transcript(
        self,
        *,
        stats: dict,
        segment_count: int,
    ) -> str:
        """Best-effort label for why STT returned no text (for logs/metrics)."""
        duration = float(stats.get('duration_seconds') or 0.0)
        if duration < 0.2:
            return 'window_too_short'
        mean_rms = stats.get('mean_rms_dbfs')
        if mean_rms is not None and mean_rms < self._low_energy_dbfs_threshold:
            return 'low_energy'
        if self._speech_ratio(stats) < 0.01:
            return 'mostly_silent'
        if self._vad_filter and segment_count == 0:
            return 'empty_segments_vad_on'
        if not self._vad_filter and segment_count == 0:
            return 'empty_segments_vad_off'
        return 'no_text_other'

    def _get_model(self) -> Any:
        """Lazily create and cache the faster-whisper model instance."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    'faster-whisper is not installed. '
                    'Install project requirements before running transcription.'
                ) from error

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

        return self._model

    def _import_numpy(self) -> Any:
        """Import numpy lazily to avoid hard runtime dependency during module import."""
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                'numpy is required for transcription audio preprocessing.'
            ) from error
        return np

    def _calculate_confidence(self, segments: list[Any]) -> float:
        """Estimate a confidence score from faster-whisper segments."""
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
