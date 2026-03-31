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
    'faster-whisper transcription time for a window (ms).',
    buckets=(50, 100, 250, 500, 800, 1200, 2000, 4000, 8000),
)

ANALYSIS_MS = _metric_or_noop(
    Histogram,
    'analysis_ms',
    'Text analysis time for a window after transcription (ms).',
    buckets=(5, 20, 50, 100, 200, 300, 500, 1000),
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

FEEDBACK_PUBLISH_ERRORS_TOTAL = _metric_or_noop(
    Counter,
    'feedback_publish_errors_total',
    'Total feedback publish errors in Python gRPC client.',
)

ANALYSIS_SUPPRESSED_BY_QUALITY_TOTAL = _metric_or_noop(
    Counter,
    'analysis_suppressed_by_quality_total',
    'Total analyzed windows that had one or more signal families suppressed for precision.',
)

ANALYSIS_FULL_SIGNAL_TOTAL = _metric_or_noop(
    Counter,
    'analysis_full_signal_total',
    'Total analyzed windows processed with full signal validity.',
)

