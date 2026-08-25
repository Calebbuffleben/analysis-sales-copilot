"""Transcription and text-analysis orchestration for ready audio windows."""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any, Callable, Optional

from ...feedback_trace import log_feedback_trace, make_feedback_trace_id
from ...pipeline_latency import LatencyTraceContext, log_speech_to_publish_ms
from ..audio_buffer.audio_diagnostics import compute_pcm_window_stats
from ..backend_feedback.publish_dispatcher import PublishDispatcher
from ..backend_feedback.types import BackendFeedbackEvent
from ..text_analysis.text_analysis_service import TextAnalysisService
from ..text_analysis.types import TextAnalysisResult, TranscriptionChunk

from ...metrics.realtime_metrics import (
    ANALYSIS_MS,
    ACOUSTIC_CLASS_TOTAL,
    ACOUSTIC_ROUTING_SKIPPED_TOTAL,
    AUDIO_LLM_INPUT_SECONDS_TOTAL,
    AUDIO_LLM_SILENCE_SKIPPED_TOTAL,
    PIPELINE_TOTAL_MS,
    STT_MS,
    WINDOW_END_TO_PIPELINE_START_MS,
    WINDOW_PROCESSED_TOTAL,
    WINDOW_SKIPPED_EMPTY_TOTAL,
)
from ..acoustic_fingerprint.correlation_metrics import CORRELATION_METRICS
from .live_host_observe import should_skip_live_host_observe
from .partial_turn_coordinator import FinalAction
from .utterance_completeness import text_fingerprint

logger = logging.getLogger(__name__)


def _is_host_role(role: object) -> bool:
    return str(role or '').strip().lower() == 'host'


def _normalize_acoustic_class(value: object) -> str:
    normalized = str(value or '').strip().lower()
    if normalized in {'seller', 'customer', 'unknown'}:
        return normalized
    return ''


def _resolve_routing_class(
    *,
    acoustic_class: object = '',
    extra_meta: Optional[dict[str, object]] = None,
    routing_enabled: bool = True,
    shadow_mode: bool = False,
    record_metrics: bool = True,
) -> str:
    """Return class used for routing; empty means ignore acoustic routing."""
    meta = extra_meta or {}
    cls = _normalize_acoustic_class(
        acoustic_class or meta.get('acoustic_class') or meta.get('acousticClass'),
    )
    if not cls:
        return ''
    if record_metrics:
        mode = 'shadow' if shadow_mode else 'active'
        ACOUSTIC_CLASS_TOTAL.labels(acoustic_class=cls, mode=mode).inc()
        CORRELATION_METRICS.observe_window(
            acoustic_class=cls,
            confidence=float(
                meta.get('correlation_confidence')
                or meta.get('correlationConfidence')
                or 0.0,
            ),
        )
    if not routing_enabled:
        if record_metrics:
            ACOUSTIC_ROUTING_SKIPPED_TOTAL.labels(reason='disabled').inc()
        return ''
    if shadow_mode:
        if record_metrics:
            ACOUSTIC_ROUTING_SKIPPED_TOTAL.labels(reason='shadow').inc()
        return ''
    return cls


def _should_observe_only(
    *,
    participant_role: object,
    acoustic_class: object = '',
    extra_meta: Optional[dict[str, object]] = None,
    routing_enabled: bool = True,
    shadow_mode: bool = False,
    record_metrics: bool = True,
) -> bool:
    """Host mic OR loopback classified as seller → context only, no feedback."""
    if _is_host_role(participant_role):
        return True
    cls = _resolve_routing_class(
        acoustic_class=acoustic_class,
        extra_meta=extra_meta,
        routing_enabled=routing_enabled,
        shadow_mode=shadow_mode,
        record_metrics=record_metrics,
    )
    return cls == 'seller'


def _should_suppress_partial_publish(
    *,
    participant_role: object,
    acoustic_class: object = '',
    extra_meta: Optional[dict[str, object]] = None,
    routing_enabled: bool = True,
    shadow_mode: bool = False,
    record_metrics: bool = False,
) -> bool:
    """Partials on seller/unknown loopback must not publish feedback."""
    if _is_host_role(participant_role):
        return True
    cls = _resolve_routing_class(
        acoustic_class=acoustic_class,
        extra_meta=extra_meta,
        routing_enabled=routing_enabled,
        shadow_mode=shadow_mode,
        record_metrics=record_metrics,
    )
    return cls in {'seller', 'unknown'}


class TranscriptionPipelineService:
    """Orchestrate STT, text analysis and feedback publishing."""

    def __init__(
        self,
        transcription_service: Optional[Any],
        text_analysis_service: TextAnalysisService,
        publish_dispatcher: PublishDispatcher,
        default_language: Optional[str] = None,
        *,
        partial_coordinator: Optional[Any] = None,
        partial_min_confidence: float = 0.7,
        feedback_allow_host_publish: bool = False,
        acoustic_routing_enabled: bool = True,
        acoustic_shadow_mode: bool = False,
        multimodal_audio_enabled: bool = False,
        live_host_context_fn: Optional[Callable[[str, str], None]] = None,
        live_host_observe_interval_ms: int = 15_000,
        metrics_host_fn: Optional[Callable[..., None]] = None,
    ) -> None:
        self._transcription_service = transcription_service
        self._text_analysis_service = text_analysis_service
        self._publish_dispatcher = publish_dispatcher
        self._default_language = self._normalize_language(default_language)
        self._stream_language_hints: dict[str, str] = {}
        self._partial_coordinator = partial_coordinator
        self._partial_min_confidence = partial_min_confidence
        self._feedback_allow_host_publish = feedback_allow_host_publish
        self._acoustic_routing_enabled = acoustic_routing_enabled
        self._acoustic_shadow_mode = acoustic_shadow_mode
        self._multimodal_audio_enabled = multimodal_audio_enabled
        self._live_host_context_fn = live_host_context_fn
        self._live_host_observe_interval_ms = max(0, int(live_host_observe_interval_ms))
        self._live_host_last_observe_ms: dict[str, int] = {}
        self._metrics_host_fn = metrics_host_fn

    def _observe_host_talk_stats(self, window_pcm: bytes, meta: dict) -> None:
        if self._metrics_host_fn is None or not _is_host_role(meta.get('participant_role')):
            return
        try:
            stats = compute_pcm_window_stats(
                window_pcm,
                sample_rate=int(meta.get('sample_rate', 0) or 0),
                channels=max(int(meta.get('channels', 1) or 1), 1),
            )
            samples = max(int(stats.get('samples_count') or 0), 1)
            speech_ratio = float(stats.get('speech_count') or 0) / samples
            duration_ms = int(float(stats.get('duration_seconds') or 0) * 1000)
            self._metrics_host_fn(
                tenant_id=str(meta.get('tenant_id') or ''),
                meeting_id=str(meta.get('meeting_id') or ''),
                duration_ms=duration_ms,
                speech_ratio=speech_ratio,
            )
        except Exception:
            logger.debug('host talk stats hook failed', exc_info=True)

    def _should_skip_live_host_observe(self, meeting_id: str) -> bool:
        """Throttle host generateContent while Live owns the client path."""
        return should_skip_live_host_observe(
            live_host_context_enabled=self._live_host_context_fn is not None,
            interval_ms=self._live_host_observe_interval_ms,
            last_observe_ms=self._live_host_last_observe_ms.get(meeting_id, 0),
            now_ms=int(time.time() * 1000),
        )

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
        self._observe_host_talk_stats(window_pcm, meta)
        if self._multimodal_audio_enabled:
            self._process_audio_window(stream_key, window_pcm, meta)
            return
        if self._transcription_service is None:
            logger.debug(
                'Ignoring ready window because local STT is disabled | stream_key=%s',
                stream_key,
            )
            return
        t_pipeline_start = time.perf_counter()
        t_wall_pipeline_start_ms = int(time.time() * 1000)
        logger.info(f"[Step 1] Início do processamento da janela de áudio (stream={stream_key})")
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

        t_stt_start = time.perf_counter()
        transcription = self._transcription_service.transcribe(window_pcm, enriched_meta)
        t_stt_end = time.perf_counter()

        logger.info(f"[Step 2] Transcrição concluída: '{transcription.text}'")
        STT_MS.observe((t_stt_end - t_stt_start) * 1000.0)
        if not transcription.text.strip():
            WINDOW_SKIPPED_EMPTY_TOTAL.inc()
            skip_msg = (
                '⏭️ Pipeline skip (empty transcript) | stream_key=%s | reason=%s | '
                'vad_filter=%s | segments=%s | language=%s | fallback_language=%s | '
                'stt_ms=%.1f | total_ms=%.1f'
            )
            skip_args = (
                stream_key,
                transcription.empty_reason,
                transcription.vad_filter_used,
                transcription.segment_count,
                enriched_meta.get('language'),
                fallback_language,
                (t_stt_end - t_stt_start) * 1000.0,
                (t_stt_end - t_pipeline_start) * 1000.0,
            )
            if transcription.empty_reason == 'low_energy':
                logger.debug(skip_msg, *skip_args)
            else:
                logger.info(skip_msg, *skip_args)
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
            tenant_id=str(enriched_meta.get('tenant_id') or ''),
            participant_role=str(enriched_meta.get('participant_role') or ''),
        )
        logger.info(f"[Step 3] Enviando transcrição para análise do Gemini")
        t_ana_start = time.perf_counter()
        observe_only = _should_observe_only(
            participant_role=chunk.participant_role,
            acoustic_class=enriched_meta.get('acoustic_class'),
            extra_meta=enriched_meta,
            routing_enabled=self._acoustic_routing_enabled,
            shadow_mode=self._acoustic_shadow_mode,
        )
        if observe_only:
            analysis = self._text_analysis_service.observe_context(chunk)
        else:
            analysis = self._text_analysis_service.analyze(chunk)
        self._apply_audio_window_stats(analysis, window_pcm, enriched_meta)
        t_ana_end = time.perf_counter()
        ANALYSIS_MS.observe((t_ana_end - t_ana_start) * 1000.0)
        t_pub_start = time.perf_counter()
        published_enqueued = (
            False if observe_only else self._handle_transcript(stream_key, chunk, analysis)
        )
        t_pub_end = time.perf_counter()

        if published_enqueued:
            WINDOW_PROCESSED_TOTAL.inc()

        PIPELINE_TOTAL_MS.observe((t_pub_end - t_pipeline_start) * 1000.0)
        stt_ms = (t_stt_end - t_stt_start) * 1000.0
        analysis_ms = (t_ana_end - t_ana_start) * 1000.0
        enqueue_ms = (t_pub_end - t_pub_start) * 1000.0
        total_ms = (t_pub_end - t_pipeline_start) * 1000.0
        tid = make_feedback_trace_id(
            chunk.meeting_id,
            chunk.participant_id,
            chunk.window_end_ms,
        )
        log_feedback_trace(
            logger,
            logging.INFO,
            'python.pipeline',
            trace_id=tid,
            meeting_id=chunk.meeting_id,
            participant_id=chunk.participant_id,
            window_end_ms=chunk.window_end_ms,
            extra={
                'streamKey': stream_key,
                'queueWaitMs': queue_wait_ms,
                'windowEndToPipelineStartMs': window_end_to_pipeline_start_ms,
                'publishEnqueued': published_enqueued,
                'sttMs': round(stt_ms, 1),
                'analysisMs': round(analysis_ms, 1),
                'enqueueMs': round(enqueue_ms, 1),
                'totalMs': round(total_ms, 1),
                'hasDirectFeedback': bool(analysis.direct_feedback),
                'contextOnly': observe_only,
                'acousticClass': _normalize_acoustic_class(
                    enriched_meta.get('acoustic_class'),
                ),
                'transcriptChars': len(chunk.text or ''),
            },
        )

    def _process_audio_window(
        self,
        stream_key: str,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Send one bounded PCM window directly to the multimodal LLM."""
        stats = compute_pcm_window_stats(
            window_pcm,
            sample_rate=int(meta.get('sample_rate', 0) or 0),
            channels=max(int(meta.get('channels', 1) or 1), 1),
        )
        samples = max(int(stats.get('samples_count') or 0), 1)
        speech_ratio = float(stats.get('speech_count') or 0) / samples
        mean_rms = stats.get('mean_rms_dbfs')
        if speech_ratio <= 0.0 and (mean_rms is None or float(mean_rms) < -55.0):
            WINDOW_SKIPPED_EMPTY_TOTAL.inc()
            AUDIO_LLM_SILENCE_SKIPPED_TOTAL.inc()
            return

        chunk = TranscriptionChunk(
            meeting_id=str(meta['meeting_id']),
            participant_id=str(meta['participant_id']),
            track=str(meta['track']),
            text='',
            confidence=0.0,
            timestamp_ms=int(meta['window_end_ms']),
            window_start_ms=int(meta['window_start_ms']),
            window_end_ms=int(meta['window_end_ms']),
            tenant_id=str(meta.get('tenant_id') or ''),
            participant_role=str(meta.get('participant_role') or ''),
        )
        observe_only = _should_observe_only(
            participant_role=chunk.participant_role,
            acoustic_class=meta.get('acoustic_class'),
            extra_meta=meta,
            routing_enabled=self._acoustic_routing_enabled,
            shadow_mode=self._acoustic_shadow_mode,
        )
        wav_bytes = self._pcm_to_wav(
            window_pcm,
            sample_rate=int(meta.get('sample_rate', 0) or 0),
            channels=max(int(meta.get('channels', 1) or 1), 1),
        )
        AUDIO_LLM_INPUT_SECONDS_TOTAL.inc(
            len(window_pcm)
            / max(
                int(meta.get('sample_rate', 0) or 0)
                * max(int(meta.get('channels', 1) or 1), 1)
                * 2,
                1,
            ),
        )
        started = time.perf_counter()
        if observe_only:
            if self._should_skip_live_host_observe(chunk.meeting_id):
                logger.info(
                    'live.host_observe_skipped | meeting=%s | interval_ms=%s',
                    chunk.meeting_id,
                    self._live_host_observe_interval_ms,
                )
                return
            analysis, evidence = self._text_analysis_service.observe_audio_context(
                chunk,
                wav_bytes,
            )
            if self._live_host_context_fn is not None:
                self._live_host_last_observe_ms[chunk.meeting_id] = int(
                    time.time() * 1000,
                )
                summary = (
                    f'estado={analysis.conversation_state_json[:500]} '
                    f'evidence={evidence[:200]}'
                )
                try:
                    self._live_host_context_fn(chunk.meeting_id, summary)
                except Exception:
                    logger.exception(
                        'Live host context inject failed | meeting=%s',
                        chunk.meeting_id,
                    )
        else:
            analysis, evidence = self._text_analysis_service.analyze_audio(
                chunk,
                wav_bytes,
            )
        ANALYSIS_MS.observe((time.perf_counter() - started) * 1000.0)
        chunk.text = evidence
        self._apply_audio_window_stats(analysis, window_pcm, meta)
        published = False if observe_only else self._handle_transcript(
            stream_key,
            chunk,
            analysis,
        )
        if published:
            WINDOW_PROCESSED_TOTAL.inc()
        log_feedback_trace(
            logger,
            logging.INFO,
            'python.pipeline',
            trace_id=make_feedback_trace_id(
                chunk.meeting_id,
                chunk.participant_id,
                chunk.window_end_ms,
            ),
            meeting_id=chunk.meeting_id,
            participant_id=chunk.participant_id,
            window_end_ms=chunk.window_end_ms,
            extra={
                'streamKey': stream_key,
                'provider': 'gemini_audio',
                'publishEnqueued': published,
                'hasDirectFeedback': bool(analysis.direct_feedback),
                'contextOnly': observe_only,
                'evidenceChars': len(evidence),
                'speechRatio': round(speech_ratio, 4),
            },
        )

    @staticmethod
    def _pcm_to_wav(
        pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> bytes:
        if sample_rate <= 0:
            raise ValueError('sample_rate must be positive')
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return output.getvalue()

    def process_transcript(
        self,
        stream_key: str,
        chunk: TranscriptionChunk,
        audio_stats: Optional[dict[str, object]] = None,
        *,
        transcript_source: str = 'final',
        extra_meta: Optional[dict[str, object]] = None,
    ) -> None:
        """Process a streaming transcript from a cloud STT provider."""
        if not chunk.text.strip():
            WINDOW_SKIPPED_EMPTY_TOTAL.inc()
            logger.debug(
                '⏭️ Pipeline skip (empty streaming transcript) | stream_key=%s',
                stream_key,
            )
            return

        source = (transcript_source or 'final').strip().lower()
        meta = dict(extra_meta or {})
        coordinator = self._partial_coordinator

        if source == 'final' and coordinator is not None:
            action = coordinator.plan_final_action(stream_key, chunk.text)
            if action == FinalAction.SKIP:
                logger.info(
                    'Skipping final analyze/publish; partial already published | '
                    'stream_key=%s | meeting=%s',
                    stream_key,
                    chunk.meeting_id,
                )
                coordinator.reset_turn(stream_key)
                return
            if action == FinalAction.RECONCILE:
                partial_conf = coordinator.partial_published_confidence(stream_key)
                logger.info(
                    'Final reconcile: partial text diverged; re-analyzing | '
                    'stream_key=%s | meeting=%s | partialConfidence=%.2f',
                    stream_key,
                    chunk.meeting_id,
                    partial_conf,
                )

        t_pipeline_start = time.perf_counter()
        t_wall_pipeline_start_ms = int(time.time() * 1000)
        window_end_to_pipeline_start_ms = max(
            0,
            t_wall_pipeline_start_ms - int(chunk.window_end_ms or 0),
        )
        WINDOW_END_TO_PIPELINE_START_MS.observe(
            float(window_end_to_pipeline_start_ms),
        )

        if source == 'partial':
            logger.info("[Step 2] Partial transcript stable: '%s'", chunk.text)
        else:
            logger.info(
                "[Step 2] Streaming transcript finalized: '%s'",
                chunk.text,
            )
        logger.info("[Step 3] Enviando transcrição para análise do Gemini")
        t_ana_start = time.perf_counter()
        observe_only = _should_observe_only(
            participant_role=chunk.participant_role,
            acoustic_class=meta.get('acoustic_class'),
            extra_meta=meta,
            routing_enabled=self._acoustic_routing_enabled,
            shadow_mode=self._acoustic_shadow_mode,
        )
        publish_host = (
            _is_host_role(chunk.participant_role)
            and self._feedback_allow_host_publish
            and _normalize_acoustic_class(meta.get('acoustic_class')) != 'seller'
        )
        suppress_partial = _should_suppress_partial_publish(
            participant_role=chunk.participant_role,
            acoustic_class=meta.get('acoustic_class'),
            extra_meta=meta,
            routing_enabled=self._acoustic_routing_enabled,
            shadow_mode=self._acoustic_shadow_mode,
        )

        if observe_only and not publish_host:
            analysis = self._text_analysis_service.observe_context(chunk)
        else:
            analysis = self._text_analysis_service.analyze(chunk)
        self._apply_streaming_audio_stats(analysis, audio_stats or {})
        t_ana_end = time.perf_counter()
        ANALYSIS_MS.observe((t_ana_end - t_ana_start) * 1000.0)

        t_pub_start = time.perf_counter()
        published_enqueued = False
        if source == 'partial':
            if suppress_partial:
                logger.info(
                    'Partial publish suppressed by acoustic class | '
                    'stream_key=%s | acousticClass=%s',
                    stream_key,
                    _normalize_acoustic_class(meta.get('acoustic_class')),
                )
            elif (
                analysis.direct_feedback
                and analysis.confidence >= self._partial_min_confidence
            ):
                published_enqueued = self._handle_transcript(
                    stream_key,
                    chunk,
                    analysis,
                )
                if published_enqueued and coordinator is not None:
                    coordinator.note_partial_published(
                        stream_key,
                        chunk.text,
                        analysis.confidence,
                    )
            else:
                logger.info(
                    'Partial analysis produced no publishable feedback | '
                    'stream_key=%s | hasFeedback=%s | confidence=%.2f',
                    stream_key,
                    bool(analysis.direct_feedback),
                    analysis.confidence,
                )
        elif publish_host or not observe_only:
            published_enqueued = self._handle_transcript(
                stream_key,
                chunk,
                analysis,
            )
        t_pub_end = time.perf_counter()
        tid = make_feedback_trace_id(
            chunk.meeting_id,
            chunk.participant_id,
            chunk.window_end_ms,
        )

        if published_enqueued:
            WINDOW_PROCESSED_TOTAL.inc()
            if source == 'partial':
                speech_to_publish = log_speech_to_publish_ms(
                    logger,
                    LatencyTraceContext(
                        trace_id=tid,
                        meeting_id=chunk.meeting_id,
                        participant_id=chunk.participant_id,
                        window_end_ms=chunk.window_end_ms,
                    ),
                    partial_stable_wall_ms=int(chunk.window_end_ms),
                    transcript_source='partial',
                )
                meta['speechToPublishMs'] = speech_to_publish

        if source == 'final' and coordinator is not None:
            coordinator.reset_turn(stream_key)

        PIPELINE_TOTAL_MS.observe((t_pub_end - t_pipeline_start) * 1000.0)
        analysis_ms = (t_ana_end - t_ana_start) * 1000.0
        enqueue_ms = (t_pub_end - t_pub_start) * 1000.0
        total_ms = (t_pub_end - t_pipeline_start) * 1000.0
        log_feedback_trace(
            logger,
            logging.INFO,
            'python.pipeline',
            trace_id=tid,
            meeting_id=chunk.meeting_id,
            participant_id=chunk.participant_id,
            window_end_ms=chunk.window_end_ms,
            extra={
                'streamKey': stream_key,
                'provider': 'assemblyai',
                'transcriptSource': source,
                'textFingerprint': text_fingerprint(chunk.text),
                'windowEndToPipelineStartMs': window_end_to_pipeline_start_ms,
                'publishEnqueued': published_enqueued,
                'sttMs': 0.0,
                'analysisMs': round(analysis_ms, 1),
                'enqueueMs': round(enqueue_ms, 1),
                'totalMs': round(total_ms, 1),
                'hasDirectFeedback': bool(analysis.direct_feedback),
                'contextOnly': observe_only and not publish_host,
                'acousticClass': _normalize_acoustic_class(meta.get('acoustic_class')),
                'transcriptChars': len(chunk.text or ''),
                **{
                    key: meta[key]
                    for key in (
                        'completenessReason',
                        'turnOpenMs',
                        'speechToPublishMs',
                    )
                    if key in meta
                },
            },
        )

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
            participant_role=transcript.participant_role or None,
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=transcript.timestamp_ms,
            window_start_ms=transcript.window_start_ms,
            window_end_ms=transcript.window_end_ms,
            message='Text analysis ingress event',
            transcript_text=transcript.text,
            transcript_confidence=transcript.confidence,
            analysis=analysis,
            tenant_id=transcript.tenant_id,
        )

    def _apply_audio_window_stats(
        self,
        analysis: TextAnalysisResult,
        window_pcm: bytes,
        meta: dict,
    ) -> None:
        """Attach audio-window stats used by backend feedback rules."""
        channels = max(int(meta.get('channels', 1)), 1)
        sample_rate = int(meta.get('sample_rate', 0) or 0)
        stats = compute_pcm_window_stats(
            window_pcm,
            sample_rate=sample_rate,
            channels=channels,
        )
        analysis.samples_count = int(stats.get('samples_count') or 0)
        analysis.speech_count = int(stats.get('speech_count') or 0)
        mean = stats.get('mean_rms_dbfs')
        analysis.mean_rms_dbfs = mean if mean is None else float(mean)

    def _apply_streaming_audio_stats(
        self,
        analysis: TextAnalysisResult,
        audio_stats: dict[str, object],
    ) -> None:
        """Attach provider-side audio stats to the analysis payload."""
        samples = audio_stats.get('samples_count')
        speech = audio_stats.get('speech_count')
        mean = audio_stats.get('mean_rms_dbfs')
        if samples is not None:
            analysis.samples_count = int(samples)
        if speech is not None:
            analysis.speech_count = int(speech)
        if mean is not None:
            analysis.mean_rms_dbfs = float(mean)

    def _normalize_language(self, language: Optional[object]) -> Optional[str]:
        if language is None:
            return None
        value = str(language).strip().lower()
        return value or None
