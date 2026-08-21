"""TTL cache of tenant playbook templates for Live (fetch off hot path)."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

FetchFn = Callable[[str], list[dict[str, Any]]]


class PlaybookCatalogCache:
    """In-memory catalog per tenant. Ceiling: process-local; upgrade: Redis."""

    def __init__(
        self,
        *,
        backend_http_base_url: str = '',
        bootstrap_key: str = '',
        service_jwt_provider: Any = None,
        ttl_sec: float = 60.0,
        fetch_fn: Optional[FetchFn] = None,
        redis_client: Any = None,
    ) -> None:
        self._base = (backend_http_base_url or '').rstrip('/')
        self._bootstrap_key = (bootstrap_key or '').strip()
        self._jwt_provider = service_jwt_provider
        self._ttl_sec = max(1.0, float(ttl_sec))
        self._fetch_fn = fetch_fn
        self._redis = redis_client
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def get(self, tenant_id: str) -> list[dict[str, Any]]:
        tid = (tenant_id or '').strip()
        if not tid:
            return []
        now = time.time()
        with self._lock:
            hit = self._entries.get(tid)
            if hit is not None and (now - hit[0]) < self._ttl_sec:
                return list(hit[1])
        redis_key = f'playbooks:catalog:{tid}'
        if self._redis is not None:
            try:
                raw = self._redis.get(redis_key)
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        with self._lock:
                            self._entries[tid] = (now, data)
                        return list(data)
            except Exception:
                logger.exception('playbook.catalog_redis_get_failed | tenant=%s', tid)
        templates = self._fetch(tid)
        with self._lock:
            self._entries[tid] = (time.time(), templates)
        if self._redis is not None and templates:
            try:
                self._redis.setex(redis_key, int(self._ttl_sec), json.dumps(templates))
            except Exception:
                logger.exception('playbook.catalog_redis_set_failed | tenant=%s', tid)
        return list(templates)

    def get_cached(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return in-memory catalog only — never HTTP (WS hot path)."""
        tid = (tenant_id or '').strip()
        if not tid:
            return []
        with self._lock:
            hit = self._entries.get(tid)
            if hit is None:
                return []
            return list(hit[1])

    def get_by_key(self, tenant_id: str, *, hot_path: bool = False) -> dict[str, dict[str, Any]]:
        templates = self.get_cached(tenant_id) if hot_path else self.get(tenant_id)
        out: dict[str, dict[str, Any]] = {}
        for t in templates:
            key = str(t.get('key') or '').strip()
            if key:
                out[key] = t
        return out

    def warm(self, tenant_id: str) -> None:
        try:
            n = len(self.get(tenant_id))
            logger.info(
                'playbook.catalog_loaded | tenant=%s | n=%s',
                tenant_id,
                n,
            )
        except Exception:
            logger.exception(
                'playbook.catalog_warm_failed | tenant=%s',
                tenant_id,
            )

    def _fetch(self, tenant_id: str) -> list[dict[str, Any]]:
        if self._fetch_fn is not None:
            try:
                return list(self._fetch_fn(tenant_id) or [])
            except Exception:
                logger.exception('playbook.catalog_fetch_fn_failed | tenant=%s', tenant_id)
                return []
        if not self._base:
            return []
        url = (
            f'{self._base}/internal/playbooks/catalog?'
            + urllib.parse.urlencode({'tenantId': tenant_id})
        )
        headers: dict[str, str] = {'Accept': 'application/json'}
        if self._bootstrap_key:
            headers['x-service-bootstrap-key'] = self._bootstrap_key
        elif self._jwt_provider is not None:
            try:
                token = self._jwt_provider.get_token()
                if token:
                    headers['Authorization'] = f'Bearer {token}'
            except Exception:
                logger.exception('playbook.catalog_jwt_failed')
                return []
        else:
            return []

        req = urllib.request.Request(url, headers=headers, method='GET')
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                body = resp.read().decode('utf-8')
            data = json.loads(body)
            templates = data.get('templates') if isinstance(data, dict) else None
            if not isinstance(templates, list):
                return []
            return [t for t in templates if isinstance(t, dict)]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                'playbook.catalog_fetch_failed | tenant=%s | error=%s',
                tenant_id,
                exc,
            )
            return []
        except Exception:
            logger.exception('playbook.catalog_fetch_failed | tenant=%s', tenant_id)
            return []
