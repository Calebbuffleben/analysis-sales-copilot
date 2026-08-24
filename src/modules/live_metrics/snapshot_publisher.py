"""Async snapshot publisher — never blocks the Live turn path."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

THROTTLE_MS = 3000
QUEUE_MAX = 32


class SnapshotPublisher:
    """Bounded latest-wins queue. Ceiling: one worker thread, drop oldest."""

    def __init__(
        self,
        publish_fn: Callable[[dict[str, Any]], None],
        *,
        throttle_ms: int = THROTTLE_MS,
    ) -> None:
        self._publish_fn = publish_fn
        self._throttle_ms = max(50, int(throttle_ms))
        self._lock = threading.Lock()
        self._pending: deque[dict[str, Any]] = deque(maxlen=QUEUE_MAX)
        self._last_sent_ms: dict[str, int] = {}
        self._last_band: dict[str, str] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker = threading.Thread(
            target=self._loop,
            name='monitor-snapshot-publisher',
            daemon=True,
        )
        self._worker.start()

    def enqueue(self, snapshot: dict[str, Any], *, force: bool = False) -> None:
        item = dict(snapshot)
        if force:
            item['_force'] = True
        key = f"{item.get('tenant_id')}:{item.get('meeting_id')}"
        band = item.get('health_band') or ''
        now = int(time.time() * 1000)
        with self._lock:
            last = self._last_sent_ms.get(key, 0)
            prev_band = self._last_band.get(key)
            band_changed = bool(band) and band != prev_band
            self._pending = deque(
                (pending for pending in self._pending
                 if f"{pending.get('tenant_id')}:{pending.get('meeting_id')}" != key),
                maxlen=QUEUE_MAX,
            )
            if not force and not band_changed and now - last < self._throttle_ms:
                self._pending.append(item)
                self._wake.set()
                return
            self._pending.append(item)
            self._wake.set()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = None
            with self._lock:
                if self._pending:
                    item = self._pending.popleft()
            if item is None:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            key = f"{item.get('tenant_id')}:{item.get('meeting_id')}"
            force = bool(item.get('_force'))
            band = str(item.get('health_band') or '')
            now = int(time.time() * 1000)
            with self._lock:
                last = self._last_sent_ms.get(key, 0)
                prev_band = self._last_band.get(key)
                others = [
                    pending for pending in self._pending
                    if f"{pending.get('tenant_id')}:{pending.get('meeting_id')}" != key
                ]
            band_changed = bool(band) and band != prev_band
            if not force and not band_changed and now - last < self._throttle_ms:
                with self._lock:
                    has_newer = any(
                        f"{pending.get('tenant_id')}:{pending.get('meeting_id')}" == key
                        for pending in self._pending
                    )
                    if not has_newer:
                        self._pending.append(item)
                if not others:
                    remaining = (self._throttle_ms - (now - last)) / 1000.0
                    self._wake.wait(min(0.3, max(0.05, remaining)))
                    self._wake.clear()
                continue
            try:
                self._publish_fn(item)
                with self._lock:
                    self._last_sent_ms[key] = int(time.time() * 1000)
                    if band:
                        self._last_band[key] = band
            except Exception:
                logger.exception(
                    'monitor snapshot publish failed | meeting=%s',
                    item.get('meeting_id'),
                )


def snapshot_to_json_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        'tenant_id': snapshot.get('tenant_id') or '',
        'meeting_id': snapshot.get('meeting_id') or '',
        'health_score': int(snapshot.get('health_score') or 50),
        'talk_listen_json': json.dumps(snapshot.get('talk_listen') or {}, ensure_ascii=False),
        'objections_json': json.dumps(snapshot.get('objections') or {}, ensure_ascii=False),
        'playbook_adherence_json': json.dumps(
            snapshot.get('playbook_adherence') or {},
            ensure_ascii=False,
        ),
        'sentiment_trend_json': json.dumps(snapshot.get('sentiment') or {}, ensure_ascii=False),
        'alerts_json': json.dumps(snapshot.get('alerts') or [], ensure_ascii=False),
        'ts_ms': int(snapshot.get('ts_ms') or time.time() * 1000),
    }
