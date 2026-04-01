"""
PublishDispatcher decouples STT/analysis workers from downstream network I/O.

Pipeline behavior:
- STT workers enqueue backend publish events immediately (non-blocking).
- Dedicated publish workers call gRPC with bounded retries.
- If the publish queue is full or the event is stale, events are dropped
  to protect realtime latency guarantees.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from ...metrics.realtime_metrics import (
    PUBLISH_DROPPED_TOTAL,
    PUBLISH_ENQUEUED_TOTAL,
    PUBLISH_QUEUE_SIZE,
    PUBLISH_GRPC_MS,
    WINDOW_END_TO_PUBLISH_ACK_MS,
    WINDOW_END_TO_PUBLISH_ENQUEUE_MS,
)
from .types import BackendFeedbackEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PublishItem:
    event: BackendFeedbackEvent
    enqueued_at_ms: int
    attempts: int


PublishFn = Callable[[BackendFeedbackEvent], Optional[float]]


class PublishDispatcher:
    """
    Bounded queue with worker threads for backend gRPC publishes.
    """

    def __init__(
        self,
        publish_fn: PublishFn,
        *,
        max_queue_size: int = 64,
        worker_threads: int = 2,
        max_event_age_ms: int = 10_000,
        retry_limit: int = 1,
        retry_backoff_ms: int = 200,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError('max_queue_size must be >= 1')
        if worker_threads < 1:
            raise ValueError('worker_threads must be >= 1')
        if retry_limit < 0:
            raise ValueError('retry_limit must be >= 0')
        if retry_backoff_ms < 0:
            raise ValueError('retry_backoff_ms must be >= 0')

        self._publish_fn = publish_fn
        self._max_queue_size = max_queue_size
        self._max_event_age_ms = max_event_age_ms
        self._retry_limit = retry_limit
        self._retry_backoff_ms = retry_backoff_ms

        self._queue: Deque[_PublishItem] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False
        self._workers: list[threading.Thread] = []

        for i in range(worker_threads):
            t = threading.Thread(
                target=self._worker_loop,
                name=f'publish-worker-{i}',
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def enqueue(self, event: BackendFeedbackEvent) -> bool:
        """
        Non-blocking enqueue.

        Returns:
        - True if accepted into the publish queue
        - False if dropped (full queue or stale event)
        """
        now_ms = int(time.time() * 1000)
        event_age_ms = now_ms - int(event.window_end_ms or 0)
        if event_age_ms > self._max_event_age_ms:
            PUBLISH_DROPPED_TOTAL.inc()
            logger.warning(
                'publish drop (stale at enqueue) | meetingId=%s | participantId=%s | '
                'window_end_ms=%s | event_age_ms=%s | max_event_age_ms=%s',
                event.meeting_id,
                event.participant_id,
                event.window_end_ms,
                event_age_ms,
                self._max_event_age_ms,
            )
            return False

        item = _PublishItem(
            event=event,
            enqueued_at_ms=now_ms,
            attempts=0,
        )

        with self._lock:
            if self._shutdown:
                return False
            if len(self._queue) >= self._max_queue_size:
                PUBLISH_DROPPED_TOTAL.inc()
                logger.warning(
                    'publish drop (queue full) | meetingId=%s | participantId=%s | '
                    'window_end_ms=%s | queue_len=%s | max=%s',
                    event.meeting_id,
                    event.participant_id,
                    event.window_end_ms,
                    len(self._queue),
                    self._max_queue_size,
                )
                return False

            self._queue.append(item)
            PUBLISH_ENQUEUED_TOTAL.inc()
            PUBLISH_QUEUE_SIZE.set(len(self._queue))
            WINDOW_END_TO_PUBLISH_ENQUEUE_MS.observe(
                float(now_ms - int(event.window_end_ms or 0)),
            )
            self._not_empty.notify()
            return True

    def get_queue_size(self) -> int:
        """Current number of queued publish items."""
        with self._lock:
            return len(self._queue)

    def get_max_queue_size(self) -> int:
        """Configured max queue capacity (backpressure / bounded queue)."""
        return self._max_queue_size

    def _worker_loop(self) -> None:
        while True:
            with self._not_empty:
                while not self._shutdown and len(self._queue) == 0:
                    self._not_empty.wait(timeout=0.5)
                if self._shutdown and len(self._queue) == 0:
                    return
                if not self._queue:
                    continue
                item = self._queue.popleft()
                PUBLISH_QUEUE_SIZE.set(len(self._queue))

            self._process_item(item)

    def _process_item(self, item: _PublishItem) -> None:
        now_ms = int(time.time() * 1000)
        event_age_ms = now_ms - int(item.event.window_end_ms or 0)
        if event_age_ms > self._max_event_age_ms:
            PUBLISH_DROPPED_TOTAL.inc()
            logger.warning(
                'publish drop (stale at worker) | meetingId=%s | participantId=%s | '
                'window_end_ms=%s | event_age_ms=%s | max_event_age_ms=%s',
                item.event.meeting_id,
                item.event.participant_id,
                item.event.window_end_ms,
                event_age_ms,
                self._max_event_age_ms,
            )
            return

        t0 = time.perf_counter()
        publish_grpc_ms = None
        had_exception = False
        try:
            publish_grpc_ms = self._publish_fn(item.event)
        except Exception:
            # The underlying client already logs errors and increments error counters.
            had_exception = True
            logger.exception(
                'publish dispatcher failed | meetingId=%s participantId=%s windowEndMs=%s',
                item.event.meeting_id,
                item.event.participant_id,
                item.event.window_end_ms,
            )

        t1 = time.perf_counter()
        publish_ms_fallback = (t1 - t0) * 1000.0
        publish_ms = (
            float(publish_grpc_ms) if publish_grpc_ms is not None else publish_ms_fallback
        )
        PUBLISH_GRPC_MS.observe(publish_ms)

        t_wall_ack_end_ms = int(time.time() * 1000)
        ack_age_ms = t_wall_ack_end_ms - int(item.event.window_end_ms or 0)
        WINDOW_END_TO_PUBLISH_ACK_MS.observe(float(ack_age_ms))

        if publish_grpc_ms is None:
            # Only retry when we observed an exception. If the client is disabled
            # it may return None without raising, and retry would be wasteful.
            if had_exception and item.attempts < self._retry_limit:
                next_item = _PublishItem(
                    event=item.event,
                    enqueued_at_ms=item.enqueued_at_ms,
                    attempts=item.attempts + 1,
                )
                if self._retry_backoff_ms > 0:
                    time.sleep(self._retry_backoff_ms / 1000.0)
                self._requeue_with_latest(next_item)
            return

    def _requeue_with_latest(self, item: _PublishItem) -> None:
        # Requeue as best-effort. If queue is full, drop rather than block.
        with self._lock:
            if self._shutdown:
                return
            if len(self._queue) >= self._max_queue_size:
                PUBLISH_DROPPED_TOTAL.inc()
                return
            self._queue.append(item)
            PUBLISH_QUEUE_SIZE.set(len(self._queue))
            self._not_empty.notify()

