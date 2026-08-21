"""TTL catalog of published specialists (HTTP to Nest, optional Redis)."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .decorator import load_builtins
from .types import SpecialistDef

logger = logging.getLogger(__name__)


class SpecialistCatalog:
    """Merge code builtins with published DB specialists."""

    def __init__(
        self,
        *,
        backend_http_base_url: str = '',
        bootstrap_key: str = '',
        service_jwt_provider: Any = None,
        ttl_sec: float = 60.0,
        redis_client: Any = None,
    ) -> None:
        self._base = (backend_http_base_url or '').rstrip('/')
        self._bootstrap_key = (bootstrap_key or '').strip()
        self._jwt_provider = service_jwt_provider
        self._ttl_sec = max(1.0, float(ttl_sec))
        self._redis = redis_client
        self._lock = threading.Lock()
        self._cached: tuple[float, dict[str, SpecialistDef]] = (0.0, {})
        self._builtins = load_builtins()

    def all(self) -> dict[str, SpecialistDef]:
        now = time.time()
        with self._lock:
            ts, items = self._cached
            if now - ts < self._ttl_sec and items:
                return dict(items)
        merged = dict(self._builtins)
        for spec in self._fetch_remote():
            if spec.key in merged and merged[spec.key].source == 'code':
                merged[spec.key].enabled = spec.enabled
                continue
            merged[spec.key] = spec
        with self._lock:
            self._cached = (time.time(), merged)
        return dict(merged)

    def enabled_for(self, selected: tuple[str, ...]) -> list[SpecialistDef]:
        catalog = self.all()
        if selected:
            keys = [k for k in selected if k in catalog]
        else:
            keys = [k for k, spec in catalog.items() if spec.enabled]
        specs = [catalog[k] for k in keys if catalog[k].enabled]
        specs.sort(key=lambda s: s.priority)
        return specs

    def register_builtins_with_backend(self) -> None:
        if not self._base:
            return
        payload = {
            'specialists': [
                {
                    'key': spec.key,
                    'name': spec.name,
                    'description': spec.description,
                    'instructions': spec.instructions,
                    'triggerPhases': list(spec.trigger_phases),
                    'triggerKeywords': list(spec.trigger_keywords),
                    'model': spec.model,
                    'maxLatencyMs': spec.max_latency_ms,
                    'priority': spec.priority,
                }
                for spec in self._builtins.values()
            ],
        }
        url = f'{self._base}/internal/specialists/register-builtins'
        try:
            self._http_json(url, method='POST', body=payload)
            logger.info(
                'specialist.builtins_registered | n=%s',
                len(self._builtins),
            )
        except Exception:
            logger.exception('specialist.builtins_register_failed')

    def _fetch_remote(self) -> list[SpecialistDef]:
        cache_key = 'specialists:catalog'
        if self._redis is not None:
            try:
                raw = self._redis.get(cache_key)
                if raw:
                    data = json.loads(raw)
                    return [self._from_wire(item) for item in data if isinstance(item, dict)]
            except Exception:
                logger.exception('specialist.catalog_redis_get_failed')
        if not self._base:
            return []
        url = f'{self._base}/internal/specialists/catalog'
        try:
            data = self._http_json(url, method='GET')
        except Exception:
            logger.warning('specialist.catalog_fetch_failed')
            return []
        items = data.get('specialists') if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        if self._redis is not None:
            try:
                self._redis.setex(cache_key, int(self._ttl_sec), json.dumps(items))
            except Exception:
                logger.exception('specialist.catalog_redis_set_failed')
        return [self._from_wire(item) for item in items if isinstance(item, dict)]

    def _http_json(
        self,
        url: str,
        *,
        method: str,
        body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self._bootstrap_key:
            headers['x-service-bootstrap-key'] = self._bootstrap_key
        elif self._jwt_provider is not None:
            token = self._jwt_provider.get_token()
            if token:
                headers['Authorization'] = f'Bearer {token}'
        payload = json.dumps(body).encode('utf-8') if body is not None else None
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            raw = resp.read().decode('utf-8')
        parsed = json.loads(raw) if raw else {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _from_wire(item: dict[str, Any]) -> SpecialistDef:
        phases = item.get('triggerPhases') or item.get('trigger_phases') or []
        keywords = item.get('triggerKeywords') or item.get('trigger_keywords') or []
        return SpecialistDef(
            key=str(item.get('key') or ''),
            name=str(item.get('name') or ''),
            description=str(item.get('description') or ''),
            instructions=str(item.get('instructions') or ''),
            tone=str(item.get('tone') or ''),
            example_message=str(item.get('exampleMessage') or item.get('example_message') or ''),
            trigger_phases=tuple(str(p) for p in phases),
            trigger_keywords=tuple(str(k) for k in keywords),
            min_confidence=float(item.get('minConfidence') or item.get('min_confidence') or 0.6),
            cooldown_sec=int(item.get('cooldownSec') or item.get('cooldown_sec') or 15),
            priority=int(item.get('priority') or 100),
            model=str(item.get('model') or 'gemini-2.5-flash'),
            max_latency_ms=int(item.get('maxLatencyMs') or item.get('max_latency_ms') or 4000),
            source='code' if str(item.get('source') or '') == 'code' else 'custom',
            enabled=bool(item.get('enabled', True)),
            icon=str(item.get('icon') or ''),
            color=str(item.get('color') or ''),
        )
