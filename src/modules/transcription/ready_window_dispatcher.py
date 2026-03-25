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
    Thread-safe bounded deque with worker threads.

    - Drops stale windows (age since window_end_ms).
    - Under pressure, drops low-priority (low speech_ratio) items first.
    - If still full, drops oldest to admit new work (log at warning).
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
        self._deque: Deque[ReadyWindowItem] = deque()
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
        """Enqueue a ready window. Returns False if dropped before queue."""
        now_ms = int(time.time() * 1000)
        window_end_ms = int(meta.get('window_end_ms', 0) or 0)
        if window_end_ms and (now_ms - window_end_ms) > self._max_age_ms:
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
            self._admit_under_pressure(item)
            self._not_empty.notify()

        logger.debug(
            'ready-window enqueued | stream_key=%s | queue_len=%s',
            stream_key,
            len(self._deque),
        )
        return True

    def _admit_under_pressure(self, item: ReadyWindowItem) -> None:
        while len(self._deque) >= self._max_size:
            oldest = self._deque[0]
            if self._is_low_priority(oldest):
                self._deque.popleft()
                logger.info(
                    '⏭️ Window drop (backlog, low priority) | stream_key=%s | speech_ratio=%s',
                    oldest.stream_key,
                    oldest.meta.get('speech_ratio'),
                )
                continue
            if self._is_low_priority(item):
                logger.info(
                    '⏭️ Window drop (backlog full, reject new low priority) | stream_key=%s | speech_ratio=%s',
                    item.stream_key,
                    item.meta.get('speech_ratio'),
                )
                return
            # Drop oldest to make room for higher-value new item
            dropped = self._deque.popleft()
            logger.warning(
                '⏭️ Window drop (backlog full, evict oldest) | stream_key=%s | speech_ratio=%s',
                dropped.stream_key,
                dropped.meta.get('speech_ratio'),
            )

        self._deque.append(item)

    def _is_low_priority(self, item: ReadyWindowItem) -> bool:
        try:
            sr = float(item.meta.get('speech_ratio', 0.0) or 0.0)
        except (TypeError, ValueError):
            sr = 0.0
        return sr < self._low_pri_below

    def _worker_loop(self) -> None:
        while True:
            with self._not_empty:
                while not self._shutdown and len(self._deque) == 0:
                    self._not_empty.wait(timeout=0.5)
                if self._shutdown and len(self._deque) == 0:
                    return
                if not self._deque:
                    continue
                item = self._deque.popleft()

            dequeue_ms = int(time.time() * 1000)
            wait_ms = dequeue_ms - item.enqueued_at_ms
            window_end_ms = int(item.meta.get('window_end_ms', 0) or 0)
            if window_end_ms and (dequeue_ms - window_end_ms) > self._max_age_ms:
                logger.info(
                    '⏭️ Window drop (stale at worker) | stream_key=%s | age_since_end_ms=%s',
                    item.stream_key,
                    dequeue_ms - window_end_ms,
                )
                continue

            logger.info(
                '📥 Window dequeue | stream_key=%s | queue_wait_ms=%s | window_end_ms=%s',
                item.stream_key,
                wait_ms,
                window_end_ms,
            )
            try:
                self._process(item.stream_key, item.window_pcm, item.meta)
            except Exception:
                logger.exception(
                    'ready-window process failed | stream_key=%s',
                    item.stream_key,
                )
