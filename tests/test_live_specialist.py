"""Tests for combined specialist analysis and secondary feedback policy."""

from __future__ import annotations

import json
import threading
import time

from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.live_feedback_publisher import LiveFeedbackPublisher
from src.modules.text_analysis.live_specialist import (
    GeminiSpecialistAnalyzer,
    LiveSpecialistRunner,
    SpecialistResult,
    SpecialistSnapshot,
)


class _FakeDispatcher:
    def __init__(self) -> None:
        self.events: list[BackendFeedbackEvent] = []

    def enqueue(self, event: BackendFeedbackEvent) -> bool:
        self.events.append(event)
        return True


def _snapshot(turn_id: str = 'turn-1', speech_end_ms: int | None = None):
    return SpecialistSnapshot(
        tenant_id='tenant-1',
        meeting_id='meeting-1',
        participant_id='remote',
        participant_role='client',
        turn_id=turn_id,
        speech_end_ms=speech_end_ms or int(time.time() * 1000),
        evidence_text='Está caro.',
        primary_feedback='Pergunte sobre o impacto do custo.',
        conversation_state={'fase_spin': 'problema'},
        host_context='Produto SaaS.',
    )


def _result(turn_id: str = 'turn-1', feedback: str = 'Valide a objeção.'):
    return SpecialistResult(
        source_turn_id=turn_id,
        fase_spin='problema',
        objecoes_detectadas=['preco', 'inventada'],
        objection_hint='Pergunte qual faixa cabe no orçamento.',
        compliance_flagged=False,
        secondary_feedback=feedback,
        secondary_feedback_type='objection',
        confidence=0.9,
        evidence_text='Está caro.',
    )


def test_specialist_result_filters_categories_and_merges_state() -> None:
    result = _result()
    assert result.objecoes_detectadas == ['preco']
    merged = result.merged_state({'objecoes_detectadas': ['tempo']})
    assert merged['objecoes_detectadas'] == ['tempo', 'preco']
    assert 'Objeção:' in result.next_turn_hint()


def test_specialist_analyzer_makes_one_json_call(monkeypatch) -> None:
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = _result().model_dump()
        return json.dumps(payload), 'sdk_developer'

    monkeypatch.setattr(
        'src.modules.text_analysis.live_specialist.generate_with_transport_chain',
        generate,
    )
    analyzer = GeminiSpecialistAnalyzer(api_key='test-key', model_name='test-model')
    result = analyzer.analyze(_snapshot())

    assert result.secondary_feedback == 'Valide a objeção.'
    assert len(calls) == 1


def test_runner_is_non_blocking_and_delivers_result() -> None:
    delivered = threading.Event()

    def analyze(snapshot):
        time.sleep(0.1)
        return _result(snapshot.turn_id)

    runner = LiveSpecialistRunner(
        analyze,
        lambda _snapshot, _result: delivered.set(),
        timeout_ms=1000,
    )
    started = time.perf_counter()
    assert runner.enqueue(_snapshot())
    enqueue_ms = (time.perf_counter() - started) * 1000.0

    assert enqueue_ms < 50
    assert delivered.wait(timeout=2.0)
    runner.shutdown()


def test_runner_latest_wins_for_pending_meeting() -> None:
    release = threading.Event()
    delivered: list[str] = []

    def analyze(snapshot):
        if snapshot.turn_id == 'turn-1':
            release.wait(timeout=1.0)
        return _result(snapshot.turn_id)

    runner = LiveSpecialistRunner(
        analyze,
        lambda snapshot, _result: delivered.append(snapshot.turn_id),
        timeout_ms=2000,
    )
    now_ms = int(time.time() * 1000)
    runner.enqueue(_snapshot('turn-1', now_ms))
    time.sleep(0.05)
    runner.enqueue(_snapshot('turn-2', now_ms + 1))
    runner.enqueue(_snapshot('turn-3', now_ms + 2))
    release.set()
    time.sleep(0.25)
    runner.shutdown()

    assert 'turn-2' not in delivered
    assert delivered[-1] == 'turn-3'


def test_secondary_publisher_applies_dedupe_and_metadata() -> None:
    dispatcher = _FakeDispatcher()
    publisher = LiveFeedbackPublisher(
        dispatcher,
        secondary_cooldown_ms=0,
        secondary_max_age_ms=10_000,
    )
    snapshot = _snapshot()
    result = _result()
    kwargs = dict(
        meeting_id=snapshot.meeting_id,
        tenant_id=snapshot.tenant_id,
        participant_id=snapshot.participant_id,
        participant_role=snapshot.participant_role,
        parent_turn_id=snapshot.turn_id,
        speech_end_ms=snapshot.speech_end_ms,
        feedback=result.secondary_feedback,
        confidence=result.confidence,
        feedback_type=result.secondary_feedback_type,
        evidence_text=result.evidence_text,
        state=result.merged_state(snapshot.conversation_state),
        specialist_metadata=result.metadata(),
    )

    assert publisher.publish_secondary_feedback(**kwargs)
    assert publisher.publish_secondary_feedback(**kwargs) is False
    assert len(dispatcher.events) == 1
    event = dispatcher.events[0]
    assert event.metadata['tier'] == 'secondary'
    assert event.metadata['parentTurnId'] == 'turn-1'
    assert event.turn_id == 'turn-1:specialist'


def test_secondary_publisher_applies_cooldown() -> None:
    dispatcher = _FakeDispatcher()
    publisher = LiveFeedbackPublisher(
        dispatcher,
        secondary_cooldown_ms=60_000,
        secondary_max_age_ms=10_000,
    )
    snapshot = _snapshot()
    common = dict(
        meeting_id=snapshot.meeting_id,
        tenant_id=snapshot.tenant_id,
        participant_id=snapshot.participant_id,
        participant_role=snapshot.participant_role,
        parent_turn_id=snapshot.turn_id,
        speech_end_ms=snapshot.speech_end_ms,
        confidence=0.9,
        feedback_type='objection',
        evidence_text='Está caro.',
        state={},
        specialist_metadata={},
    )

    assert publisher.publish_secondary_feedback(feedback='Primeiro.', **common)
    assert (
        publisher.publish_secondary_feedback(feedback='Segundo diferente.', **common)
        is False
    )
