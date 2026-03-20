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

from .types import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Turns a PCM window into transcript text using faster-whisper."""

    def __init__(
        self,
        model_size: str = 'small',
        device: str = 'cpu',
        compute_type: str = 'int8',
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[Any] = None

    def transcribe(self, window_pcm: bytes, meta: dict) -> TranscriptionResult:
        """Transcribe a single PCM audio window.

        Args:
            window_pcm: Raw PCM bytes for the current window.
            meta: Context (e.g. sample_rate, channels, meeting_id, participant_id).

        Returns:
            A `TranscriptionResult` with text and confidence.
        """
        if not window_pcm:
            return TranscriptionResult(text='', confidence=0.0)

        np = self._import_numpy()
        audio = np.frombuffer(window_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        model = self._get_model()
        language = meta.get('language')
        segments, info = model.transcribe(
            audio=audio,
            language=language or None,
            beam_size=1,
            vad_filter=True,
        )

        segment_list = list(segments)
        transcript_text = ' '.join(
            segment.text.strip()
            for segment in segment_list
            if getattr(segment, 'text', '').strip()
        ).strip()

        confidence = self._calculate_confidence(segment_list)
        detected_language = getattr(info, 'language', None)

        logger.info(
            '📝 Transcription completed | meetingId=%s | participantId=%s | chars=%s | confidence=%.3f',
            meta.get('meeting_id'),
            meta.get('participant_id'),
            len(transcript_text),
            confidence,
        )

        return TranscriptionResult(
            text=transcript_text,
            confidence=confidence,
            language=detected_language,
        )

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
