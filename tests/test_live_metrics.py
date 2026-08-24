"""Deterministic live-metrics: talk-to-listen, health score, alerts."""

from __future__ import annotations

import time

from src.modules.live_metrics.aggregator import MeetingMetricsAggregator
from src.modules.live_metrics.health_score import compute_health_score, playbook_adherence
from src.modules.live_metrics.snapshot_publisher import SnapshotPublisher
from src.modules.live_metrics.talk_stats import MONOLOGUE_MS, TalkStatsStore
from src.modules.text_analysis.live_feedback_publisher import LiveFeedbackPublisher


def test_health_score_drops_on_objections_and_falling_sentiment() -> None:
    score, factors = compute_health_score(
        estado={
            'interesse': 'baixo',
            'engajamento': 'baixo',
            'resistencia': 'alta',
            'objecoes_ativas': ['preco', 'tempo'],
            'alerta_risco_spin': True,
            'sentimento_cliente': 'negativo',
            'sentimento_tendencia': 'caindo',
        },
        host_ratio=0.8,
        hesitation_hint=True,
    )
    assert score < 40
    assert any('objeção' in f for f in factors)


def test_health_score_healthy_call() -> None:
    score, _factors = compute_health_score(
        estado={
            'interesse': 'alto',
            'engajamento': 'alto',
            'resistencia': 'baixa',
            'sentimento_cliente': 'positivo',
            'sentimento_tendencia': 'subindo',
        },
        host_ratio=0.4,
    )
    assert score >= 70


def test_playbook_adherence_from_spin() -> None:
    result = playbook_adherence({'fase_spin': 'problema'})
    assert result['percent'] == 50
    assert result['steps'][1]['done'] is True
    assert result['steps'][2]['done'] is False


def test_talk_stats_host_ratio_and_yellow_alert() -> None:
    store = TalkStatsStore()
    store.observe_host('t', 'm', duration_ms=10_000, speech_ratio=0.9, now_ms=1_000)
    store.observe_host('t', 'm', duration_ms=10_000, speech_ratio=0.9, now_ms=3_000)
    stats = store.observe_customer('t', 'm', duration_ms=2_000)
    assert stats.host_ratio > 0.7
    stats.host_monologue_ms = MONOLOGUE_MS
    msg = store.pop_yellow_alert('t', 'm')
    assert msg is not None
    assert store.pop_yellow_alert('t', 'm') is None


def test_monologue_accumulates_speech_time_and_resets_on_customer() -> None:
    store = TalkStatsStore()
    # 3 minutes of continuous host speech (silence windows contribute ~0ms).
    stats = store.observe_host('t', 'm', duration_ms=90_000, speech_ratio=1.0, now_ms=1_000)
    stats = store.observe_host('t', 'm', duration_ms=90_000, speech_ratio=1.0, now_ms=91_000)
    assert stats.host_monologue_ms >= MONOLOGUE_MS
    # A pure-silence window must not grow the monologue counter.
    stats = store.observe_host('t', 'm', duration_ms=5_000, speech_ratio=0.0, now_ms=96_000)
    assert stats.host_monologue_ms == 180_000
    # Any customer speech resets it.
    stats = store.observe_customer('t', 'm', duration_ms=1_000, now_ms=97_000)
    assert stats.host_monologue_ms == 0


def test_recent_host_ratio_uses_moving_window() -> None:
    store = TalkStatsStore()
    # Old talkative stretch, outside the 2-minute window.
    store.observe_host('t', 'm', duration_ms=60_000, speech_ratio=1.0, now_ms=10_000)
    # Recent balanced exchange.
    store.observe_host('t', 'm', duration_ms=10_000, speech_ratio=1.0, now_ms=200_000)
    stats = store.observe_customer('t', 'm', duration_ms=10_000, now_ms=205_000)
    assert stats.host_ratio > 0.8  # accumulated still dominated by the old stretch
    assert abs(stats.recent_host_ratio(now_ms=205_000) - 0.5) < 0.01


def test_host_window_keeps_last_conversation_state() -> None:
    published: list[dict] = []
    publisher = SnapshotPublisher(published.append, throttle_ms=50)
    agg = MeetingMetricsAggregator(publisher)
    estado = {
        'interesse': 'alto',
        'engajamento': 'alto',
        'resistencia': 'baixa',
        'objecoes_ativas': ['preco'],
        'sentimento_cliente': 'neutro',
        'sentimento_tendencia': 'estavel',
    }
    agg.observe_turn(
        tenant_id='t',
        meeting_id='m',
        estado=estado,
        feedback_type='opportunity',
        confidence=0.8,
    )
    agg.observe_host_window(
        tenant_id='t',
        meeting_id='m',
        duration_ms=800,
        speech_ratio=0.4,
    )
    time.sleep(0.2)
    publisher.shutdown()
    assert published
    last = published[-1]
    assert last['objections']['active'] == ['preco']


def test_observe_turn_metrics_runs_even_without_feedback_text() -> None:
    """Snapshots must update even when the tool call carries no coaching tip."""

    class _Dispatcher:
        def enqueue(self, _event) -> bool:
            return True

    published: list[dict] = []
    sink = SnapshotPublisher(published.append, throttle_ms=50)
    agg = MeetingMetricsAggregator(sink)
    live_publisher = LiveFeedbackPublisher(_Dispatcher(), metrics=agg)
    live_publisher.observe_turn_metrics(
        meeting_id='m',
        tenant_id='t',
        args={
            'feedback': '',
            'confidence': 0.0,
            'estado': {'sentimento_cliente': 'negativo'},
        },
    )
    time.sleep(0.2)
    sink.shutdown()
    assert published
    assert published[-1]['sentiment']['current'] == 'negativo'


def test_red_alert_requires_objection_and_falling_sentiment() -> None:
    published: list[dict] = []
    publisher = SnapshotPublisher(published.append, throttle_ms=50)
    agg = MeetingMetricsAggregator(publisher)
    agg.observe_turn(
        tenant_id='t',
        meeting_id='m',
        estado={
            'sentimento_cliente': 'negativo',
            'sentimento_tendencia': 'caindo',
            'objecoes_ativas': ['preco'],
        },
        feedback_type='objection',
        confidence=0.9,
    )
    time.sleep(0.2)
    publisher.shutdown()
    alerts = published[-1]['alerts']
    assert any(a['kind'] == 'red' for a in alerts)


def test_snapshot_publisher_throttles_then_force_sends() -> None:
    published: list[dict] = []
    publisher = SnapshotPublisher(published.append, throttle_ms=250)
    base = {
        'tenant_id': 't',
        'meeting_id': 'm',
        'health_score': 70,
        'health_band': 'green',
    }
    publisher.enqueue(base)
    time.sleep(0.12)
    assert len(published) == 1
    publisher.enqueue({**base, 'health_score': 71})
    time.sleep(0.05)
    assert len(published) == 1
    publisher.enqueue({**base, 'health_band': 'red', 'health_score': 20}, force=True)
    time.sleep(0.12)
    publisher.shutdown()
    assert any(item.get('health_band') == 'red' for item in published)
    assert len(published) >= 2
