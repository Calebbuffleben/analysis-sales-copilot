"""Transcription and text-analysis orchestration for ready audio windows."""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional

from ..audio_buffer.audio_diagnostics import compute_pcm_window_stats
from ..backend_feedback.publish_dispatcher import PublishDispatcher
from ..backend_feedback.types import BackendFeedbackEvent
from ..text_analysis.text_analysis_service import TextAnalysisService
from ..text_analysis.types import TextAnalysisResult, TranscriptionChunk
from .execution_profile import ExecutionProfile
from ...metrics.realtime_metrics import (
    ANALYSIS_MS,
    PIPELINE_TOTAL_MS,
    STT_MS,
    WINDOW_END_TO_PIPELINE_START_MS,
    WINDOW_PROCESSED_TOTAL,
    WINDOW_SKIPPED_EMPTY_TOTAL,
)
from .transcription_service import TranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionPipelineService:
    """Orchestrate STT, text analysis and feedback publishing."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        text_analysis_service: TextAnalysisService,
        publish_dispatcher: PublishDispatcher,
        default_language: Optional[str] = None,
    ) -> None:
        self._transcription_service = transcription_service
        self._text_analysis_service = text_analysis_service
        self._publish_dispatcher = publish_dispatcher
        self._default_language = self._normalize_language(default_language)
        self._stream_language_hints: dict[str, str] = {}
        self._profile_lock = threading.Lock()
        self._execution_profile = ExecutionProfile(
            level='L0',
            use_embeddings=True,
            semantic_pipeline_enabled=True,
            compute_category_transition=True,
            low_priority_speech_ratio_below=0.0,
            analysis_mode='full_semantic',
            signal_validity={
                'semantic_indecision': True,
                'audio_aggregate': True,
            },
        )

    def set_execution_profile(self, profile: ExecutionProfile) -> None:
        """Called by DegradationController to update execution flags."""
        with self._profile_lock:
            self._execution_profile = profile

    def _on_window_ready(
        self,
        stream_key: str,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Backward-compatible alias for SlidingWindowWorker callback."""
        self.process_window(stream_key, window_pcm, meta)

    def process_window(
        self,
        stream_key: str,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Process one ready window: STT, analysis, publish."""
        t_pipeline_start = time.perf_counter()
        t_wall_pipeline_start_ms = int(time.time() * 1000)
        enriched_meta = dict(meta)
        configured_language = self._default_language
        if configured_language:
            enriched_meta['language'] = configured_language

        fallback_language = (
            self._stream_language_hints.get(stream_key)
            if not configured_language
            else configured_language
        )
        if fallback_language:
            enriched_meta['fallback_language'] = fallback_language

        window_end_ms = int(enriched_meta.get('window_end_ms', 0) or 0)
        enqueued_at_ms = enriched_meta.get('enqueued_at_ms')
        dequeued_at_ms = enriched_meta.get('dequeued_at_ms')
        queue_wait_ms = None
        if isinstance(enqueued_at_ms, int) and isinstance(dequeued_at_ms, int):
            queue_wait_ms = max(0, dequeued_at_ms - enqueued_at_ms)

        window_end_to_pipeline_start_ms = (
            (t_wall_pipeline_start_ms - window_end_ms)
            if window_end_ms
            else None
        )

        if window_end_to_pipeline_start_ms is not None:
            WINDOW_END_TO_PIPELINE_START_MS.observe(
                float(window_end_to_pipeline_start_ms),
            )

        audio_stats = self._compute_audio_window_stats(window_pcm, enriched_meta)

        t_stt_start = time.perf_counter()
        transcription = self._transcription_service.transcribe(window_pcm, enriched_meta)
        t_stt_end = time.perf_counter()

        STT_MS.observe((t_stt_end - t_stt_start) * 1000.0)
        if not transcription.text.strip():
            WINDOW_SKIPPED_EMPTY_TOTAL.inc()
            logger.info(
                '⏭️ Pipeline skip (empty transcript) | stream_key=%s | reason=%s | '
                'vad_filter=%s | segments=%s | language=%s | fallback_language=%s | '
                'stt_ms=%.1f | total_ms=%.1f',
                stream_key,
                transcription.empty_reason,
                transcription.vad_filter_used,
                transcription.segment_count,
                enriched_meta.get('language'),
                fallback_language,
                (t_stt_end - t_stt_start) * 1000.0,
                (t_stt_end - t_pipeline_start) * 1000.0,
            )
            return

        if (
            not configured_language
            and transcription.language
            and stream_key not in self._stream_language_hints
        ):
            self._stream_language_hints[stream_key] = transcription.language
            logger.info(
                '📝 STT stream language hint learned | stream_key=%s | language=%s',
                stream_key,
                transcription.language,
            )

        chunk = TranscriptionChunk(
            meeting_id=str(enriched_meta['meeting_id']),
            participant_id=str(enriched_meta['participant_id']),
            track=str(enriched_meta['track']),
            text=transcription.text,
            confidence=transcription.confidence,
            timestamp_ms=int(enriched_meta['window_end_ms']),
            window_start_ms=int(enriched_meta['window_start_ms']),
            window_end_ms=int(enriched_meta['window_end_ms']),
        )
        t_ana_start = time.perf_counter()
        with self._profile_lock:
            execution_profile = self._execution_profile
        analysis = self._text_analysis_service.analyze(
            chunk,
            execution_profile=execution_profile,
        )
        self._apply_audio_window_stats(analysis, audio_stats)
        t_ana_end = time.perf_counter()
        ANALYSIS_MS.observe((t_ana_end - t_ana_start) * 1000.0)
        t_pub_start = time.perf_counter()
        published_enqueued = self._handle_transcript(stream_key, chunk, analysis)
        t_pub_end = time.perf_counter()

        if published_enqueued:
            WINDOW_PROCESSED_TOTAL.inc()

        PIPELINE_TOTAL_MS.observe((t_pub_end - t_pipeline_start) * 1000.0)
        logger.info(
            '⏱️ Pipeline latency | stream_key=%s | queue_wait_ms=%s | '
            'window_end_to_pipeline_start_ms=%s | publish_enqueued=%s | '
            'stt_ms=%.1f | analysis_ms=%.1f | enqueue_ms=%.1f | total_ms=%.1f',
            stream_key,
            queue_wait_ms,
            window_end_to_pipeline_start_ms,
            published_enqueued,
            (t_stt_end - t_stt_start) * 1000.0,
            (t_ana_end - t_ana_start) * 1000.0,
            (t_pub_end - t_pub_start) * 1000.0,
            (t_pub_end - t_pipeline_start) * 1000.0,
        )

    def enqueue_audio_aggregate(
        self,
        stream_key: str,
        window_pcm: bytes,
        meta: dict,
    ) -> bool:
        """Publish audio-only aggregate stats without waiting for STT."""
        with self._profile_lock:
            execution_profile = self._execution_profile
        stats = self._compute_audio_window_stats(window_pcm, meta)
        analysis = TextAnalysisResult(
            embedding=[],
            keywords=[],
            samples_count=int(stats.get('samples_count') or 0),
            speech_count=int(stats.get('speech_count') or 0),
            mean_rms_dbfs=(
                None
                if stats.get('mean_rms_dbfs') is None
                else float(stats.get('mean_rms_dbfs'))
            ),
            analysis_mode='audio_only',
            degradation_level=execution_profile.level,
            signal_validity={
                'semantic_indecision': False,
                'audio_aggregate': execution_profile.signal_validity.get(
                    'audio_aggregate',
                    True,
                ),
            },
            suppression_reasons=['semantic_not_available_for_audio_only_ingress'],
        )
        event = BackendFeedbackEvent(
            meeting_id=str(meta['meeting_id']),
            participant_id=str(meta['participant_id']),
            participant_name=None,
            participant_role=None,
            feedback_type='audio_metrics_ingress',
            severity='info',
            ts_ms=int(meta['window_end_ms']),
            window_start_ms=int(meta['window_start_ms']),
            window_end_ms=int(meta['window_end_ms']),
            message='Audio aggregate ingress event',
            transcript_text='',
            transcript_confidence=0.0,
            analysis=analysis,
        )
        try:
            return self._publish_dispatcher.enqueue(event)
        except Exception as exc:
            logger.exception(
                'Audio aggregate publish enqueue failed | stream_key=%s | error=%s',
                stream_key,
                exc,
            )
            return False

    def _handle_transcript(
        self,
        stream_key: str,
        transcript: TranscriptionChunk,
        analysis: TextAnalysisResult,
    ) -> bool:
        """Enqueue backend publish without blocking the STT worker path."""
        event = self._build_event(transcript, analysis)
        try:
            return self._publish_dispatcher.enqueue(event)
        except Exception as exc:
            logger.exception(
                'Feedback publish enqueue failed after transcript | stream_key=%s | error=%s',
                stream_key,
                exc,
            )
            return False

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

    def _compute_audio_window_stats(self, window_pcm: bytes, meta: dict) -> dict:
        """Compute PCM window stats once so text and audio feedback reuse them."""
        channels = max(int(meta.get('channels', 1)), 1)
        sample_rate = int(meta.get('sample_rate', 0) or 0)
        return compute_pcm_window_stats(
            window_pcm,
            sample_rate=sample_rate,
            channels=channels,
        )

    def _apply_audio_window_stats(
        self,
        analysis: TextAnalysisResult,
        stats: dict,
    ) -> None:
        """Attach audio-window stats used by backend feedback rules."""
        analysis.samples_count = int(stats.get('samples_count') or 0)
        analysis.speech_count = int(stats.get('speech_count') or 0)
        mean = stats.get('mean_rms_dbfs')
        analysis.mean_rms_dbfs = mean if mean is None else float(mean)

    def _normalize_language(self, language: Optional[object]) -> Optional[str]:
        if language is None:
            return None
        value = str(language).strip().lower()
        return value or None
