"""Per-meeting metrics aggregator used by Live + host windows."""

from __future__ import annotations

import logging
import time
from typing import Any

from .health_score import compute_health_score, playbook_adherence, _active_objections
from .snapshot_publisher import SnapshotPublisher
from .talk_stats import TalkStatsStore

logger = logging.getLogger(__name__)

RED_COOLDOWN_MS = 60_000


class MeetingMetricsAggregator:
    def __init__(
        self,
        publisher: SnapshotPublisher,
        *,
        catalog_cache: Any = None,
    ) -> None:
        self._talk = TalkStatsStore()
        self._publisher = publisher
        self._catalog = catalog_cache
        self._last_red_ms: dict[str, int] = {}
        self._last_turn: dict[str, dict[str, Any]] = {}

    def observe_host_window(
        self,
        *,
        tenant_id: str,
        meeting_id: str,
        duration_ms: int,
        speech_ratio: float,
    ) -> None:
        if not tenant_id or not meeting_id:
            return
        self._talk.observe_host(
            tenant_id,
            meeting_id,
            duration_ms=duration_ms,
            speech_ratio=speech_ratio,
        )
        cached = self._last_turn.get(f'{tenant_id}:{meeting_id}', {})
        self._emit(
            tenant_id,
            meeting_id,
            estado=cached.get('estado'),
            feedback_type=cached.get('feedback_type'),
            confidence=float(cached.get('confidence') or 0.0),
            energy_level=str(cached.get('energy_level') or ''),
            hesitation_hint=bool(cached.get('hesitation_hint')),
        )

    def observe_customer_turn(
        self,
        *,
        tenant_id: str,
        meeting_id: str,
        duration_ms: int,
    ) -> None:
        if not tenant_id or not meeting_id:
            return
        self._talk.observe_customer(tenant_id, meeting_id, duration_ms=duration_ms)

    def observe_turn(
        self,
        *,
        tenant_id: str,
        meeting_id: str,
        estado: dict[str, Any] | None,
        feedback_type: str | None,
        confidence: float,
        energy_level: str = '',
        hesitation_hint: bool = False,
    ) -> None:
        self._last_turn[f'{tenant_id}:{meeting_id}'] = {
            'estado': estado,
            'feedback_type': feedback_type,
            'confidence': confidence,
            'energy_level': energy_level,
            'hesitation_hint': hesitation_hint,
        }
        self._emit(
            tenant_id,
            meeting_id,
            estado=estado,
            feedback_type=feedback_type,
            confidence=confidence,
            energy_level=energy_level,
            hesitation_hint=hesitation_hint,
        )

    def _emit(
        self,
        tenant_id: str,
        meeting_id: str,
        *,
        estado: dict[str, Any] | None,
        feedback_type: str | None,
        confidence: float,
        energy_level: str = '',
        hesitation_hint: bool = False,
    ) -> None:
        now_ms = int(time.time() * 1000)
        stats = self._talk.get(tenant_id, meeting_id)
        steps = self._playbook_steps(tenant_id)
        score, factors = compute_health_score(
            estado=estado,
            host_ratio=stats.host_ratio,
            energy_level=energy_level,
            hesitation_hint=hesitation_hint,
        )
        band = 'green' if score >= 70 else 'yellow' if score >= 40 else 'red'
        alerts: list[dict[str, str]] = []
        yellow = self._talk.pop_yellow_alert(tenant_id, meeting_id)
        if yellow:
            alerts.append({'kind': 'yellow', 'message': yellow})
        red = self._maybe_red(tenant_id, meeting_id, estado, feedback_type, confidence)
        if red:
            alerts.append(red)

        snapshot = {
            'tenant_id': tenant_id,
            'meeting_id': meeting_id,
            'health_score': score,
            'health_band': band,
            'health_factors': factors,
            'talk_listen': {
                'hostSpeechMs': stats.host_speech_ms,
                'customerSpeechMs': stats.customer_speech_ms,
                'hostRatio': round(stats.host_ratio, 3),
                'hostRatioRecent': round(stats.recent_host_ratio(now_ms), 3),
                'hostMonologueMs': stats.host_monologue_ms,
                'factors': factors,
            },
            'objections': {
                'active': _active_objections(estado or {}),
                'resolved': list(estado.get('objecoes_resolvidas') or []) if estado else [],
            },
            'playbook_adherence': playbook_adherence(estado, steps),
            'sentiment': {
                'current': str((estado or {}).get('sentimento_cliente') or 'neutro'),
                'trend': str((estado or {}).get('sentimento_tendencia') or 'estavel'),
            },
            'alerts': alerts,
            'ts_ms': now_ms,
        }
        self._publisher.enqueue(snapshot, force=bool(alerts))

    def _maybe_red(
        self,
        tenant_id: str,
        meeting_id: str,
        estado: dict[str, Any] | None,
        feedback_type: str | None,
        confidence: float,
    ) -> dict[str, str] | None:
        kind = (feedback_type or '').lower()
        trend = str((estado or {}).get('sentimento_tendencia') or '').lower()
        sentiment = str((estado or {}).get('sentimento_cliente') or '').lower()
        severe = kind in {'objection', 'risk'} and confidence >= 0.7
        falling = trend == 'caindo' or sentiment == 'negativo'
        if not (severe and falling):
            return None
        key = f'{tenant_id}:{meeting_id}'
        now = int(time.time() * 1000)
        last = self._last_red_ms.get(key, 0)
        if now - last < RED_COOLDOWN_MS:
            return None
        self._last_red_ms[key] = now
        return {
            'kind': 'red',
            'message': 'Objeção grave + sentimento do lead caindo',
        }

    def _playbook_steps(self, tenant_id: str) -> list[dict[str, Any]]:
        if self._catalog is None or not tenant_id:
            return []
        try:
            # Hot path: in-memory read only; the Live session pre-warms the
            # catalog, so we never fall back to HTTP here.
            getter = getattr(self._catalog, 'get_cached', None) or self._catalog.get
            templates = getter(tenant_id)
        except Exception:
            logger.debug('playbook catalog read failed', exc_info=True)
            return []
        if not templates:
            return []
        first = templates[0]
        steps = first.get('steps') if isinstance(first, dict) else None
        return steps if isinstance(steps, list) else []
