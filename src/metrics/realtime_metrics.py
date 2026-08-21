"""
Prometheus metrics for the realtime audio->feedback pipeline.

This module is intentionally defensive:
- if `prometheus_client` is not installed (local dev/tests), metrics become no-ops
- in production, metrics are real and are exposed via `/metrics`.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover
    Counter = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]


class _NoopMetric:
    def inc(self, amount: int = 1) -> None:  # noqa: ARG002
        return

    def dec(self, amount: int = 1) -> None:  # noqa: ARG002
        return

    def set(self, value: float) -> None:  # noqa: ARG002
        return

    def observe(self, value: float) -> None:  # noqa: ARG002
        return

    def labels(self, *args: object, **kwargs: object) -> '_NoopMetric':  # noqa: ARG002
        return self


def _metric_or_noop(metric_ctor: Optional[object], *args: object, **kwargs: object) -> object:
    if metric_ctor is None:
        return _NoopMetric()
    return metric_ctor(*args, **kwargs)  # type: ignore[misc]


# --- Scheduler / Queue metrics ---

WINDOW_QUEUE_SIZE = _metric_or_noop(
    Gauge,
    'window_queue_size',
    'Current bounded ready-window queue length (Python service).',
)

WINDOW_ENQUEUED_TOTAL = _metric_or_noop(
    Counter,
    'window_enqueued_total',
    'Total ready windows enqueued for processing.',
)

WINDOW_DEQUEUED_TOTAL = _metric_or_noop(
    Counter,
    'window_dequeued_total',
    'Total ready windows dequeued by workers.',
)

QUEUE_WAIT_MS = _metric_or_noop(
    Histogram,
    'window_queue_wait_ms',
    'Time spent waiting in ready-window queue (ms).',
    buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000),
)

WINDOW_DROPPED_STALE_TOTAL = _metric_or_noop(
    Counter,
    'window_dropped_stale_total',
    'Total ready windows dropped because they became stale (too old).',
)

WINDOW_DROPPED_LOW_PRIORITY_TOTAL = _metric_or_noop(
    Counter,
    'window_dropped_low_priority_total',
    'Total ready windows dropped due to low speech_ratio under backlog pressure.',
)

WINDOW_DROPPED_BACKLOG_EVICTED_TOTAL = _metric_or_noop(
    Counter,
    'window_dropped_backlog_evicted_total',
    'Total ready windows dropped to evict the oldest window under backlog pressure.',
)


# --- Pipeline stage latency metrics ---

WINDOW_END_TO_PIPELINE_START_MS = _metric_or_noop(
    Histogram,
    'window_end_to_pipeline_start_ms',
    'Wall time from window_end_ms to start of pipeline processing (ms).',
    buckets=(10, 50, 100, 250, 500, 1000, 2000, 5000, 10000),
)

STT_MS = _metric_or_noop(
    Histogram,
    'stt_ms',
    'Local STT transcription time for a window (ms). Cloud streaming providers report their own metrics.',
    buckets=(50, 100, 250, 500, 800, 1200, 2000, 4000, 8000),
)

ANALYSIS_MS = _metric_or_noop(
    Histogram,
    'analysis_ms',
    'Text analysis time for a window after transcription (ms).',
    buckets=(5, 20, 50, 100, 200, 300, 500, 1000),
)

AUDIO_LLM_CALLS_TOTAL = _metric_or_noop(
    Counter,
    'audio_llm_calls_total',
    'Total direct audio analysis calls made to the multimodal LLM.',
)

AUDIO_LLM_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'audio_llm_errors_total',
    'Total direct audio analysis calls that failed.',
)

AUDIO_LLM_LATENCY_MS = _metric_or_noop(
    Histogram,
    'audio_llm_latency_ms',
    'Multimodal audio analysis request latency (ms).',
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 15000),
)

AUDIO_LLM_INPUT_SECONDS_TOTAL = _metric_or_noop(
    Counter,
    'audio_llm_input_seconds_total',
    'Total audio seconds sent to the multimodal LLM.',
)

AUDIO_LLM_SILENCE_SKIPPED_TOTAL = _metric_or_noop(
    Counter,
    'audio_llm_silence_skipped_total',
    'Total silent audio windows skipped before multimodal analysis.',
)

PUBLISH_GRPC_MS = _metric_or_noop(
    Histogram,
    'publish_grpc_ms',
    'gRPC PublishFeedback wall time observed by Python client (ms).',
    buckets=(1, 5, 20, 50, 100, 200, 500, 1000, 3000, 10000),
)

WINDOW_END_TO_PUBLISH_ACK_MS = _metric_or_noop(
    Histogram,
    'window_end_to_publish_ack_ms',
    'Wall time from window_end_ms to backend publish ack on Python side (ms).',
    buckets=(10, 50, 100, 250, 500, 1000, 2000, 5000, 10000),
)

WINDOW_END_TO_PUBLISH_ENQUEUE_MS = _metric_or_noop(
    Histogram,
    'window_end_to_publish_enqueue_ms',
    'Wall time from window_end_ms to publish enqueue on Python side (ms).',
    buckets=(10, 50, 100, 250, 500, 1000, 2000, 5000, 10000),
)

PUBLISH_QUEUE_SIZE = _metric_or_noop(
    Gauge,
    'publish_queue_size',
    'Current bounded publish queue length (Python service).',
)

PUBLISH_ENQUEUED_TOTAL = _metric_or_noop(
    Counter,
    'publish_enqueued_total',
    'Total feedback publish events enqueued for backend delivery.',
)

PUBLISH_DROPPED_TOTAL = _metric_or_noop(
    Counter,
    'publish_dropped_total',
    'Total feedback publish events dropped due to full queue or stale cutoff.',
)

PIPELINE_TOTAL_MS = _metric_or_noop(
    Histogram,
    'pipeline_total_ms',
    'Total pipeline time from pipeline start to publish enqueue (ms).',
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000),
)

WINDOW_PROCESSED_TOTAL = _metric_or_noop(
    Counter,
    'window_processed_total',
    'Total ready windows successfully processed and published (non-empty transcript).',
)

WINDOW_SKIPPED_EMPTY_TOTAL = _metric_or_noop(
    Counter,
    'window_skipped_empty_total',
    'Total ready windows skipped because STT returned empty transcript.',
)

AUDIO_CHUNKS_RECEIVED_TOTAL = _metric_or_noop(
    Counter,
    'audio_chunks_received_total',
    'Total gRPC audio chunks received by the Python service.',
)

AUDIO_CHUNKS_PROCESSED_TOTAL = _metric_or_noop(
    Counter,
    'audio_chunks_processed_total',
    'Total gRPC audio chunks accepted by AudioService.process_chunk.',
)

PIPELINE_QUEUE_SIZE = _metric_or_noop(
    Gauge,
    'pipeline_queue_size',
    'Coarse operational queue size for admin dashboard (active streams / backlog proxy).',
)

PIPELINE_LATENCY_MS = _metric_or_noop(
    Histogram,
    'pipeline_latency_ms',
    'End-to-end operational pipeline latency (ms) for admin dashboard.',
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 15000),
)

ASSEMBLYAI_SESSIONS_OPEN = _metric_or_noop(
    Gauge,
    'assemblyai_sessions_open',
    'Current open AssemblyAI streaming sessions.',
)

ASSEMBLYAI_SESSIONS_STARTED_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_sessions_started_total',
    'Total AssemblyAI streaming sessions started.',
)

ASSEMBLYAI_SESSIONS_TERMINATED_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_sessions_terminated_total',
    'Total AssemblyAI streaming sessions terminated.',
)

ASSEMBLYAI_AUDIO_BYTES_SENT_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_audio_bytes_sent_total',
    'Total PCM audio bytes sent to AssemblyAI streaming.',
)

ASSEMBLYAI_BILLABLE_SECONDS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_billable_seconds_total',
    'Estimated billable audio seconds sent to AssemblyAI.',
)

ASSEMBLYAI_ESTIMATED_COST_USD_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_estimated_cost_usd_total',
    'Estimated AssemblyAI cost in USD.',
)

ASSEMBLYAI_TURNS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_turns_total',
    'Total AssemblyAI turn events received.',
)

ASSEMBLYAI_FINAL_TURNS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_final_turns_total',
    'Total final AssemblyAI turns that reached end_of_turn.',
)

ASSEMBLYAI_EMPTY_TURNS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_empty_turns_total',
    'Total AssemblyAI final turns skipped because transcript was empty.',
)

ASSEMBLYAI_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_errors_total',
    'Total AssemblyAI streaming provider errors.',
)

ASSEMBLYAI_RECONNECTS_TOTAL = _metric_or_noop(
    Counter,
    'assemblyai_reconnects_total',
    'Total AssemblyAI stream reconnect attempts.',
)

ASSEMBLYAI_TURN_LATENCY_MS = _metric_or_noop(
    Histogram,
    'assemblyai_turn_latency_ms',
    'Wall time from local turn audio end to AssemblyAI final turn handling (ms).',
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

FEEDBACK_PUBLISH_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'feedback_publish_errors_total',
    'Total feedback publish errors in Python gRPC client.',
)


# --- LLM Analysis metrics ---

LLM_CALLS_TOTAL = _metric_or_noop(
    Counter,
    'llm_calls_total',
    'Total LLM API calls attempted (Gemini).',
)

LLM_CALL_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'llm_call_errors_total',
    'Total LLM API call failures (timeout, rate limit, etc.).',
)

LLM_FALLBACK_ACTIVATED_TOTAL = _metric_or_noop(
    Counter,
    'llm_fallback_activated_total',
    'Total times rule-based fallback was used (LLM failed or returned empty).',
)

LLM_FEEDBACK_EMITTED_TOTAL = _metric_or_noop(
    Counter,
    'llm_feedback_emitted_total',
    'Total feedback events generated by LLM (non-empty).',
)

LLM_CALL_DURATION_MS = _metric_or_noop(
    Histogram,
    'llm_call_duration_ms',
    'Gemini API call latency (ms).',
    buckets=(100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000),
)

GEMINI_ESTIMATED_COST_USD_TOTAL = _metric_or_noop(
    Counter,
    'gemini_estimated_cost_usd_total',
    'Estimated Gemini API cost in USD when token usage is available.',
)

GEMINI_RPM_CURRENT = _metric_or_noop(
    Gauge,
    'gemini_rpm_current',
    'Current Gemini requests per minute estimate.',
)

GEMINI_RPM_LIMIT = _metric_or_noop(
    Gauge,
    'gemini_rpm_limit',
    'Configured Gemini requests per minute limit.',
)

LLM_CONFIDENCE_SCORE = _metric_or_noop(
    Histogram,
    'llm_confidence_score',
    'LLM self-reported confidence scores (0.0-1.0).',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

LLM_CONFIDENCE_SUPPRESSED_TOTAL = _metric_or_noop(
    Counter,
    'llm_confidence_suppressed_total',
    'Total feedback suppressed due to low LLM confidence (< 0.6).',
)

LLM_CACHE_HITS_TOTAL = _metric_or_noop(
    Counter,
    'llm_cache_hits_total',
    'Total LLM response cache hits (avoided API call).',
)

LLM_CACHE_MISSES_TOTAL = _metric_or_noop(
    Counter,
    'llm_cache_misses_total',
    'Total LLM response cache misses (called API).',
)

LLM_ACTIVE_STATES = _metric_or_noop(
    Gauge,
    'llm_active_states',
    'Number of active conversation states in memory.',
)

LLM_CACHE_SIZE = _metric_or_noop(
    Gauge,
    'llm_cache_size',
    'Current number of entries in LLM response cache.',
)

LLM_CACHE_HIT_RATIO = _metric_or_noop(
    Gauge,
    'llm_cache_hit_ratio',
    'LLM response cache hit rate (0.0-1.0).',
)

LLM_RATE_LIMITED_TOTAL = _metric_or_noop(
    Counter,
    'llm_rate_limited_total',
    'Total LLM calls deferred by RPM rate limiter (queued for later dispatch).',
)

LLM_RATE_QUEUE_SIZE = _metric_or_noop(
    Gauge,
    'llm_rate_queue_size',
    'Current number of LLM analyses waiting in the RPM rate-limit queue.',
)

GEMINI_POOL_SLOTS = _metric_or_noop(
    Gauge,
    'gemini_pool_slots',
    'Number of configured Gemini API key slots.',
)

GEMINI_KEY_CALLS_TOTAL = _metric_or_noop(
    Counter,
    'gemini_key_calls_total',
    'Total Gemini API calls reserved per key slot.',
    ['slot'],
)

GEMINI_KEY_RPM_LIMITED_TOTAL = _metric_or_noop(
    Counter,
    'gemini_key_rpm_limited_total',
    'Total Gemini calls deferred because the selected key slot hit RPM.',
    ['slot'],
)

ACOUSTIC_CLASS_TOTAL = _metric_or_noop(
    Counter,
    'acoustic_class_total',
    'Acoustic class labels observed on STT turns (seller/customer/unknown).',
    ['acoustic_class', 'mode'],
)

ACOUSTIC_ROUTING_SKIPPED_TOTAL = _metric_or_noop(
    Counter,
    'acoustic_routing_skipped_total',
    'Turns where acoustic routing was disabled or shadow mode ignored class.',
    ['reason'],
)



# --- Gemini Live (realtime) metrics ---

LIVE_SESSIONS_OPEN = _metric_or_noop(
    Gauge,
    'live_sessions_open',
    'Current open Gemini Live sessions.',
)

LIVE_SESSIONS_STARTED_TOTAL = _metric_or_noop(
    Counter,
    'live_sessions_started_total',
    'Total Gemini Live sessions started.',
)

LIVE_SESSIONS_CLOSED_TOTAL = _metric_or_noop(
    Counter,
    'live_sessions_closed_total',
    'Total Gemini Live sessions closed.',
)

LIVE_SESSIONS_RESUMED_TOTAL = _metric_or_noop(
    Counter,
    'live_sessions_resumed_total',
    'Total Gemini Live session resumptions.',
)

LIVE_AUDIO_BYTES_SENT_TOTAL = _metric_or_noop(
    Counter,
    'live_audio_bytes_sent_total',
    'Total PCM audio bytes sent to Gemini Live.',
)

LIVE_TOOL_CALLS_TOTAL = _metric_or_noop(
    Counter,
    'live_tool_calls_total',
    'Total emit_feedback tool calls received from Gemini Live.',
)

LIVE_TOOL_CALLS_INVALID_TOTAL = _metric_or_noop(
    Counter,
    'live_tool_calls_invalid_total',
    'Total emit_feedback tool calls rejected by validation.',
)

LIVE_TOOL_CALLS_DEDUPED_TOTAL = _metric_or_noop(
    Counter,
    'live_tool_calls_deduped_total',
    'Total emit_feedback tool calls dropped by turnId dedupe.',
)

LIVE_UNEXPECTED_AUDIO_BYTES_TOTAL = _metric_or_noop(
    Counter,
    'live_unexpected_audio_bytes_total',
    'Total unexpected audio output bytes discarded from Gemini Live.',
)

LIVE_VAD_END_TO_TOOL_CALL_MS = _metric_or_noop(
    Histogram,
    'live_vad_end_to_tool_call_ms',
    'Wall time from speech end (activity_end) to emit_feedback tool call (ms).',
    buckets=(50, 100, 200, 400, 600, 850, 1000, 1500, 3000, 5000),
)

LIVE_STAGE_MS = _metric_or_noop(
    Histogram,
    'live_stage_ms',
    'Per-stage Live turn latency in milliseconds.',
    ('stage',),
    buckets=(5, 10, 20, 40, 80, 150, 300, 600, 1000, 2000, 4000),
)

LIVE_SPEECH_END_TO_WS_MS = _metric_or_noop(
    Histogram,
    'live_speech_end_to_ws_ms',
    'Wall time from speech end to WS broadcast enqueue (ms).',
    buckets=(50, 100, 200, 400, 600, 850, 1000, 1500, 3000, 5000),
)

LIVE_COST_USD_TOTAL = _metric_or_noop(
    Counter,
    'live_cost_usd_total',
    'Estimated Gemini Live cost in USD (token-based).',
)

LIVE_COST_USD_PER_MEETING = _metric_or_noop(
    Gauge,
    'live_cost_usd_per_meeting',
    'Estimated Gemini Live cost in USD for the most recently updated meeting.',
)

LIVE_COST_LIMIT_TRIPS_TOTAL = _metric_or_noop(
    Counter,
    'live_cost_limit_trips_total',
    'Total times a Live session hit the per-meeting cost guardrail.',
)

LIVE_FALLBACK_TOTAL = _metric_or_noop(
    Counter,
    'live_fallback_total',
    'Total times Live path fell back to generateContent multimodal.',
)

LIVE_ADMISSION_REJECTED_TOTAL = _metric_or_noop(
    Counter,
    'live_admission_rejected_total',
    'Live sessions rejected by admission control.',
    ['reason'],
)

LANGGRAPH_NODE_MS = _metric_or_noop(
    Histogram,
    'langgraph_node_ms',
    'LangGraph node execution latency in milliseconds.',
    ('node',),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
)

SPECIALIST_QUEUE_SIZE = _metric_or_noop(
    Gauge,
    'live_specialist_queue_size',
    'Current number of queued Live specialist jobs.',
)

SPECIALIST_CALLS_TOTAL = _metric_or_noop(
    Counter,
    'live_specialist_calls_total',
    'Total combined Live specialist Gemini calls started.',
)

SPECIALIST_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'live_specialist_errors_total',
    'Total Live specialist failures and timeouts.',
)

SPECIALIST_DROPPED_TOTAL = _metric_or_noop(
    Counter,
    'live_specialist_dropped_total',
    'Total Live specialist jobs or results dropped.',
    ('reason',),
)

SPECIALIST_LATENCY_MS = _metric_or_noop(
    Histogram,
    'live_specialist_latency_ms',
    'Combined Live specialist Gemini latency in milliseconds.',
    buckets=(100, 250, 500, 1000, 2000, 3000, 5000, 8000, 15000),
)

SECONDARY_FEEDBACK_PUBLISHED_TOTAL = _metric_or_noop(
    Counter,
    'live_secondary_feedback_published_total',
    'Total specialist secondary feedback events published.',
)

SECONDARY_FEEDBACK_SUPPRESSED_TOTAL = _metric_or_noop(
    Counter,
    'live_secondary_feedback_suppressed_total',
    'Total specialist secondary feedback events suppressed.',
    ('reason',),
)

SPEECH_END_TO_SECONDARY_WS_MS = _metric_or_noop(
    Histogram,
    'live_speech_end_to_secondary_ws_ms',
    'Wall time from source speech end to secondary WS broadcast enqueue.',
    buckets=(100, 250, 500, 1000, 2000, 3000, 5000, 8000, 15000),
)
