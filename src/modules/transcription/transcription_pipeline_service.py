"""Transcription and text-analysis orchestration for ready audio windows."""

from __future__ import annotations

import logging
import math

from ..backend_feedback.grpc_feedback_client import BackendFeedbackClient
from ..backend_feedback.types import BackendFeedbackEvent
from ..text_analysis.text_analysis_service import TextAnalysisService
from ..text_analysis.types import TextAnalysisResult, TranscriptionChunk
from .transcription_service import TranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionPipelineService:
    """Orchestrate STT, text analysis and feedback publishing."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        text_analysis_service: TextAnalysisService,
        backend_feedback_client: BackendFeedbackClient,
    ) -> None:
        self._transcription_service = transcription_service
        self._text_analysis_service = text_analysis_service
        self._backend_feedback_client = backend_feedback_client

    def _on_window_ready(
        self,
        stream_key: str,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Callback invoked by SlidingWindowWorker when a window is ready."""
        transcription = self._transcription_service.transcribe(window_pcm, meta)
        if not transcription.text.strip():
            return

        chunk = TranscriptionChunk(
            meeting_id=str(meta['meeting_id']),
            participant_id=str(meta['participant_id']),
            track=str(meta['track']),
            text=transcription.text,
            confidence=transcription.confidence,
            timestamp_ms=int(meta['window_end_ms']),
            window_start_ms=int(meta['window_start_ms']),
            window_end_ms=int(meta['window_end_ms']),
        )
        analysis = self._text_analysis_service.analyze(chunk)
        self._apply_audio_window_stats(analysis, window_pcm, meta)
        self._handle_transcript(stream_key, chunk, analysis)

    def _handle_transcript(
        self,
        stream_key: str,
        transcript: TranscriptionChunk,
        analysis: TextAnalysisResult,
    ) -> None:
        """Publish a raw text-analysis ingress event to the backend."""
        event = self._build_event(transcript, analysis)
        self._backend_feedback_client.publish_feedback(event)

    def _build_event(
        self,
        transcript: TranscriptionChunk,
        analysis: TextAnalysisResult,
    ) -> BackendFeedbackEvent:
        """Build a raw text-analysis ingress event for the backend."""
        return BackendFeedbackEvent(
            meeting_id=transcript.meeting_id,
            participant_id=transcript.participant_id,
            participant_name=None,
            participant_role=None,
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=transcript.timestamp_ms,
            window_start_ms=transcript.window_start_ms,
            window_end_ms=transcript.window_end_ms,
            message='Text analysis ingress event',
            transcript_text=transcript.text,
            transcript_confidence=transcript.confidence,
            analysis=analysis,
        )

    def _apply_audio_window_stats(
        self,
        analysis: TextAnalysisResult,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Attach audio-window stats used by backend feedback rules."""
        channels = max(int(meta.get('channels', 1)), 1)
        sample_count = len(window_pcm) // (2 * channels)
        if sample_count <= 0:
            analysis.samples_count = 0
            analysis.speech_count = 0
            analysis.mean_rms_dbfs = None
            return

        import array

        pcm_array = array.array('h')
        pcm_array.frombytes(window_pcm)
        if channels > 1:
            mono_samples = pcm_array[::channels]
        else:
            mono_samples = pcm_array

        analysis.samples_count = len(mono_samples)
        analysis.speech_count = sum(
            1 for sample in mono_samples if abs(sample) >= 500
        )
        if not mono_samples:
            analysis.mean_rms_dbfs = None
            return

        rms = math.sqrt(
            sum(sample * sample for sample in mono_samples) / len(mono_samples)
        )
        if rms <= 0:
            analysis.mean_rms_dbfs = -120.0
            return

        analysis.mean_rms_dbfs = round(
            20 * math.log10(rms / 32768.0),
            2,
        )
