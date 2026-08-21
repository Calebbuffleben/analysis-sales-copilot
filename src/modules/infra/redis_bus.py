"""Optional Redis for FeedbackHub pub/sub, catalogs, and meeting state."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CONV_TTL_SEC = 1800
HOST_TTL_SEC = 1800
CATALOG_TTL_SEC = 60


class RedisBus:
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        self._thread: Optional[threading.Thread] = None
        self._handlers: dict[str, Callable[[str], None]] = {}

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            logger.exception('redis.ping_failed')
            return False

    def publish(self, channel: str, payload: dict[str, Any]) -> None:
        self._client.publish(channel, json.dumps(payload, ensure_ascii=False))

    def subscribe(self, channel: str, handler: Callable[[str], None]) -> None:
        self._handlers[channel] = handler
        self._pubsub.subscribe(channel)
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._listen,
                name='redis-bus',
                daemon=True,
            )
            self._thread.start()

    def get(self, key: str) -> Optional[str]:
        value = self._client.get(key)
        return str(value) if value is not None else None

    def setex(self, key: str, ttl_sec: int, value: str) -> None:
        self._client.setex(key, ttl_sec, value)

    def _listen(self) -> None:
        for message in self._pubsub.listen():
            if message.get('type') != 'message':
                continue
            channel = str(message.get('channel') or '')
            handler = self._handlers.get(channel)
            if handler is None:
                continue
            data = message.get('data')
            if isinstance(data, str):
                handler(data)

    def close(self) -> None:
        try:
            self._pubsub.close()
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass


def try_create_redis(url: Optional[str]) -> Optional[RedisBus]:
    if not (url or '').strip():
        return None
    bus = RedisBus(url.strip())
    if not bus.ping():
        logger.warning('REDIS_URL set but ping failed — using process-local state')
        return None
    logger.info('Redis connected')
    return bus


class MeetingStateStore:
    """Conversation + host context. Redis when available, else process-local."""

    def __init__(self, redis: Optional[RedisBus] = None) -> None:
        self._redis = redis
        self._lock = threading.Lock()
        self._conv: dict[str, dict[str, Any]] = {}
        self._host: dict[str, str] = {}

    @staticmethod
    def _conv_key(tenant_id: str, meeting_id: str) -> str:
        return f'conv:{tenant_id}:{meeting_id}'

    @staticmethod
    def _host_key(tenant_id: str, meeting_id: str) -> str:
        return f'hostctx:{tenant_id}:{meeting_id}'

    def get_conversation(self, tenant_id: str, meeting_id: str) -> dict[str, Any]:
        key = self._conv_key(tenant_id, meeting_id)
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception:
                logger.exception('redis conv get failed')
        with self._lock:
            return dict(self._conv.get(key) or {})

    def set_conversation(
        self,
        tenant_id: str,
        meeting_id: str,
        state: dict[str, Any],
    ) -> None:
        key = self._conv_key(tenant_id, meeting_id)
        payload = dict(state or {})
        with self._lock:
            self._conv[key] = payload
        if self._redis is not None:
            try:
                self._redis.setex(key, CONV_TTL_SEC, json.dumps(payload, ensure_ascii=False))
            except Exception:
                logger.exception('redis conv set failed')

    def get_host_context(self, tenant_id: str, meeting_id: str) -> str:
        key = self._host_key(tenant_id, meeting_id)
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    return raw
            except Exception:
                logger.exception('redis hostctx get failed')
        with self._lock:
            return self._host.get(key) or ''

    def set_host_context(self, tenant_id: str, meeting_id: str, summary: str) -> None:
        key = self._host_key(tenant_id, meeting_id)
        text = (summary or '')[:1500]
        with self._lock:
            self._host[key] = text
        if self._redis is not None:
            try:
                self._redis.setex(key, HOST_TTL_SEC, text)
            except Exception:
                logger.exception('redis hostctx set failed')
