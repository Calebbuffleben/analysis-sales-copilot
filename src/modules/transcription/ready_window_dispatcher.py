"""Bounded queue + worker pool for ready-window processing (realtime backpressure).

SlidingWindowWorker only enqueues here; STT runs off the gRPC chunk hot path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional

from ...metrics.realtime_metrics import (
    QUEUE_WAIT_MS,
    WINDOW_DROPPED_BACKLOG_EVICTED_TOTAL,
    WINDOW_DROPPED_LOW_PRIORITY_TOTAL,
    WINDOW_DROPPED_STALE_TOTAL,
    WINDOW_DEQUEUED_TOTAL,
    WINDOW_ENQUEUED_TOTAL,
    WINDOW_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadyWindowItem:
    """One ready window to process."""

    stream_key: str
    window_pcm: bytes
    meta: dict[str, Any]
    enqueued_at_ms: int


ProcessFn = Callable[[str, bytes, dict[str, Any]], None]


class ReadyWindowDispatcher:
    """
    Stream-aware bounded scheduler with latest-wins semantics.

    Invariants per `stream_key`:
    - max_inflight = 1
    - max_pending = 1
    - if a new window arrives while one is pending/inflight, the newer window replaces it
      (inflight is never interrupted; latest-wins applies to the next pending window).

    Under global pressure:
    - drop stale windows (TTL)
    - evict low priority pending windows (by speech_ratio) first
    - otherwise evict the oldest pending window to admit new work

    Workers:
    - pick the next eligible stream in round-robin order using an internal eligibility queue
    - process at most one window per stream at a time (max_inflight=1)
    """

    def __init__(
        self,
        process_fn: ProcessFn,
        *,
        max_queue_size: int = 8,
        worker_threads: int = 2,
        max_age_ms: int = 25_000,
        low_priority_speech_ratio_below: float = 0.02,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError('max_queue_size must be >= 1')
        if worker_threads < 1:
            raise ValueError('worker_threads must be >= 1')
        self._process = process_fn
        self._max_size = max_queue_size
        self._max_age_ms = max_age_ms
        self._low_pri_below = low_priority_speech_ratio_below
        # pending windows waiting to be scheduled for each stream_key.
        # at most 1 item per stream_key.
        self._pending: dict[str, ReadyWindowItem] = {}
        # streams currently processed by a worker.
        # guarantees max_inflight=1 per stream_key.
        self._inflight: set[str] = set()
        # round-robin eligible streams: duplicates can exist; eligibility is checked on dequeue.
        self._eligible_streams: Deque[str] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False
        self._workers: list[threading.Thread] = []
        for i in range(worker_threads):
            t = threading.Thread(
                target=self._worker_loop,
                name=f'ready-window-worker-{i}',
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
        for t in self._workers:
            t.join(timeout=timeout / max(len(self._workers), 1))

    def enqueue(self, stream_key: str, window_pcm: bytes, meta: dict[str, Any]) -> bool:
        """Enqueue a ready window. Returns False if dropped before scheduling."""
        now_ms = int(time.time() * 1000)
        window_end_ms = int(meta.get('window_end_ms', 0) or 0)
        if window_end_ms and (now_ms - window_end_ms) > self._max_age_ms:
            WINDOW_DROPPED_STALE_TOTAL.inc()
            logger.info(
                '⏭️ Window drop (stale at enqueue) | stream_key=%s | '
                'age_ms=%s | max_age_ms=%s | window_end_ms=%s',
                stream_key,
                now_ms - window_end_ms,
                self._max_age_ms,
                window_end_ms,
            )
            return False

        item = ReadyWindowItem(
            stream_key=stream_key,
            window_pcm=window_pcm,
            meta=dict(meta),
            enqueued_at_ms=now_ms,
        )

        with self._lock:
            if self._shutdown:
                return False
            # Replace semantics (latest-wins):
            # - if stream is pending: replace pending item (no capacity change)
            # - if stream is inflight: set/replace pending item (capacity change only if pending didn't exist)
            if stream_key in self._pending:
                self._pending[stream_key] = item
                self._eligible_streams.append(stream_key)
                WINDOW_ENQUEUED_TOTAL.inc()
                WINDOW_QUEUE_SIZE.set(self._current_working_streams())
                self._not_empty.notify()
                return True

            # If the stream is inflight but has no pending item yet, we will add pending (capacity may change).
            will_add_pending = stream_key not in self._inflight
            # If stream is inflight => will_add_pending can be true (we still add pending), and it still counts as pending.
            # So instead of branching on inflight, base admission on whether pending exists.
            will_add_pending = stream_key not in self._pending

            if will_add_pending:
                if self._current_working_streams() >= self._max_size:
                    self._evict_or_drop_for_new_pending(item)
                    # After eviction attempt we may still be full; check again.
                    if self._current_working_streams() >= self._max_size:
                        return False

                self._pending[stream_key] = item
                # eligible order: if it's inflight, it will be re-added after inflight completes.
                # if it's not inflight, it can be picked by workers now.
                if stream_key not in self._inflight:
                    self._eligible_streams.append(stream_key)
                WINDOW_ENQUEUED_TOTAL.inc()
                WINDOW_QUEUE_SIZE.set(self._current_working_streams())
                self._not_empty.notify()
                return True

            return False

    def _current_working_streams(self) -> int:
        # bounded by max_queue_size by admission control
        return len(self._pending) + len(self._inflight)

    def _is_low_priority_item(self, item: ReadyWindowItem) -> bool:
        try:
            sr = float(item.meta.get('speech_ratio', 0.0) or 0.0)
        except (TypeError, ValueError):
            sr = 0.0
        return sr < self._low_pri_below

    def _evict_or_drop_for_new_pending(self, new_item: ReadyWindowItem) -> None:
        """
        Maintain global capacity when adding a new pending window would overflow.

        Eviction strategy:
        1) Prefer evicting low-priority pending streams.
        2) If none exist, evict the oldest pending window.
        3) If new item is low priority, drop it instead.
        """
        if not self._pending:
            # If we are full without pending items, the only occupied capacity is inflight.
            # We can't evict inflight (it would break max_inflight semantics), so drop new.
            if self._is_low_priority_item(new_item):
                WINDOW_DROPPED_LOW_PRIORITY_TOTAL.inc()
            else:
                WINDOW_DROPPED_BACKLOG_EVICTED_TOTAL.inc()
            return

        # 1) Evict an existing low-priority pending item if any.
        low_pri_keys = [k for k, v in self._pending.items() if self._is_low_priority_item(v)]
        if low_pri_keys:
            # evict oldest among low-priority
            evict_key = min(
                low_pri_keys,
                key=lambda k: int(self._pending[k].enqueued_at_ms),
            )
            self._pending.pop(evict_key, None)
            WINDOW_DROPPED_LOW_PRIORITY_TOTAL.inc()
            return

        # 2) No low-priority pending items exist.
        # If the new item is low priority, drop it.
        if self._is_low_priority_item(new_item):
            WINDOW_DROPPED_LOW_PRIORITY_TOTAL.inc()
            return

        # 3) Otherwise evict oldest pending window to admit new work.
        evict_key = min(
            self._pending.keys(),
            key=lambda k: int(self._pending[k].enqueued_at_ms),
        )
        self._pending.pop(evict_key, None)
        WINDOW_DROPPED_BACKLOG_EVICTED_TOTAL.inc()

    def _worker_loop(self) -> None:
        while True:
            with self._not_empty:
                while not self._shutdown and not self._has_eligible_work():
                    self._not_empty.wait(timeout=0.5)
                if self._shutdown and not self._has_eligible_work():
                    return
                stream_key, item = self._dequeue_next_item()
                # After dequeue, the global "working streams" reduces by 0/1 depending on inflight change,
                # but the gauge should represent inflight+pending state.
                WINDOW_DEQUEUED_TOTAL.inc()
                WINDOW_QUEUE_SIZE.set(self._current_working_streams())

            dequeue_ms = int(time.time() * 1000)
            wait_ms = dequeue_ms - item.enqueued_at_ms
            window_end_ms = int(item.meta.get('window_end_ms', 0) or 0)
            if window_end_ms and (dequeue_ms - window_end_ms) > self._max_age_ms:
                WINDOW_DROPPED_STALE_TOTAL.inc()
                logger.info(
                    '⏭️ Window drop (stale at worker) | stream_key=%s | age_since_end_ms=%s',
                    stream_key,
                    dequeue_ms - window_end_ms,
                )
                with self._lock:
                    self._inflight.discard(stream_key)
                    WINDOW_QUEUE_SIZE.set(self._current_working_streams())
                    self._not_empty.notify()
                continue

            logger.info(
                '📥 Window dequeue | stream_key=%s | queue_wait_ms=%s | window_end_ms=%s',
                stream_key,
                wait_ms,
                window_end_ms,
            )
            QUEUE_WAIT_MS.observe(float(wait_ms))
            try:
                # Keep per-window correlation timestamps without changing any
                # external payload/contract: this meta is only internal to Python.
                meta_with_queue_times = dict(item.meta)
                meta_with_queue_times['enqueued_at_ms'] = item.enqueued_at_ms
                meta_with_queue_times['dequeued_at_ms'] = dequeue_ms
                meta_with_queue_times['queue_wait_ms'] = wait_ms
                self._process(stream_key, item.window_pcm, meta_with_queue_times)
            except Exception:
                logger.exception(
                    'ready-window process failed | stream_key=%s',
                    stream_key,
                )
            finally:
                # Mark stream as no longer inflight and re-queue pending (latest-wins).
                with self._not_empty:
                    self._inflight.discard(stream_key)
                    if stream_key in self._pending:
                        self._eligible_streams.append(stream_key)
                    WINDOW_QUEUE_SIZE.set(self._current_working_streams())
                    self._not_empty.notify()

    def _has_eligible_work(self) -> bool:
        # eligibility means there exists a pending stream that is not inflight.
        # we rely on `_eligible_streams` for round-robin, but we must clean stale entries.
        if not self._pending:
            return False
        if self._eligible_streams:
            # quick check: if any eligible stream key is still pending and not inflight.
            for _ in range(len(self._eligible_streams)):
                k = self._eligible_streams[0]
                if k in self._pending and k not in self._inflight:
                    return True
                # stale key: rotate it away (keep fairness roughly)
                self._eligible_streams.rotate(-1)
            return False
        # If no eligible queue exists (should be rare), any pending stream not inflight is eligible.
        return any(k not in self._inflight for k in self._pending.keys())

    def _dequeue_next_item(self) -> tuple[str, ReadyWindowItem]:
        # Dequeue one item from the next eligible stream in round-robin order.
        while True:
            if not self._eligible_streams:
                # Should be covered by _has_eligible_work, but be defensive.
                pending_keys = [k for k in self._pending.keys() if k not in self._inflight]
                if not pending_keys:
                    raise RuntimeError('No eligible stream found despite has_eligible_work=true')
                stream_key = pending_keys[0]
            else:
                stream_key = self._eligible_streams.popleft()

            if stream_key in self._inflight:
                continue
            if stream_key not in self._pending:
                continue

            item = self._pending.pop(stream_key)
            self._inflight.add(stream_key)
            return stream_key, item
