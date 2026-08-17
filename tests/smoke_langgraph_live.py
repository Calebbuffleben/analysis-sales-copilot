"""Smoke test do runtime LangGraph no caminho Live (sem Gemini real).

Valida que a implementação sobe, compila sem checkpointer, executa os
grafos pré/pós-tool, enfileira o especialista sem bloquear e respeita as
flags de settings. Não chama API externa.

Uso:
  python tests/smoke_langgraph_live.py
  # ou
  pytest tests/smoke_langgraph_live.py -q
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.live_feedback_publisher import LiveFeedbackPublisher
from src.modules.text_analysis.live_specialist import (
    LiveSpecialistRunner,
    SpecialistResult,
    SpecialistSnapshot,
)
from src.modules.text_analysis.live_turn_graph import LiveTurnGraphs
from src.modules.text_analysis.gemini_live_session import GeminiLiveManager

# Local graph overhead budget for smoke (not the Gemini SLO).
_GRAPH_OVERHEAD_BUDGET_MS = 100.0


class _FakeDispatcher:
    def __init__(self) -> None:
        self.events: list[BackendFeedbackEvent] = []

    def enqueue(self, event: BackendFeedbackEvent) -> bool:
        self.events.append(event)
        return True


def _ok(msg: str) -> None:
    print(f'OK: {msg}')


def smoke_import_and_compile() -> LiveTurnGraphs:
    import langgraph  # noqa: F401
    from importlib.metadata import version as pkg_version

    graphs = LiveTurnGraphs()
    assert graphs.pre_tool is not None
    assert graphs.post_tool is not None
    assert graphs.pre_tool.checkpointer is None
    assert graphs.post_tool.checkpointer is None
    _ok(
        'langgraph import + graphs compiled | '
        f'version={pkg_version("langgraph")}',
    )
    return graphs


async def smoke_pre_and_post_graphs(graphs: LiveTurnGraphs) -> None:
    started = time.perf_counter()
    pre = await graphs.run_pre_tool(
        {
            'participant_role': 'client',
            'signal_valid': True,
            'base_nudge': 'base',
            'retrieve_fn': lambda: 'playbook-hit',
            'prosody_nudge': 'prosody-line',
        },
    )
    pre_ms = (time.perf_counter() - started) * 1000.0
    assert pre['route'] == 'customer'
    assert pre['nudge'] == 'base\nplaybook-hit\nprosody-line'
    assert pre_ms < _GRAPH_OVERHEAD_BUDGET_MS, f'pre-tool too slow: {pre_ms:.1f}ms'
    _ok(f'pre-tool customer path | {pre_ms:.1f}ms')

    host = await graphs.run_pre_tool(
        {
            'participant_role': 'host',
            'signal_valid': True,
            'base_nudge': 'base',
            'retrieve_fn': lambda: 'should-not-run',
        },
    )
    assert host['route'] == 'observe'
    assert 'nudge' not in host
    _ok('pre-tool host observe path')

    suppress = await graphs.run_pre_tool(
        {
            'participant_role': 'client',
            'signal_valid': False,
            'base_nudge': 'base',
        },
    )
    assert suppress['route'] == 'suppress'
    _ok('pre-tool weak-signal suppress path')

    published: list[object] = []

    async def prosody():
        return {'energy': 'high'}

    def publish(value) -> bool:
        published.append(value)
        return True

    started = time.perf_counter()
    post = await graphs.run_post_tool(
        {
            'turn_id': 'smoke-turn',
            'await_prosody_fn': prosody,
            'publish_fn': publish,
        },
    )
    post_ms = (time.perf_counter() - started) * 1000.0
    assert post['published'] is True
    assert published == [{'energy': 'high'}]
    assert post_ms < _GRAPH_OVERHEAD_BUDGET_MS, f'post-tool too slow: {post_ms:.1f}ms'
    _ok(f'post-tool publish path | {post_ms:.1f}ms')


def smoke_specialist_non_blocking() -> None:
    done = []

    def analyze(snapshot: SpecialistSnapshot) -> SpecialistResult:
        time.sleep(0.15)
        return SpecialistResult(
            source_turn_id=snapshot.turn_id,
            secondary_feedback='Smoke secondary tip',
            secondary_feedback_type='objection',
            confidence=0.9,
            evidence_text=snapshot.evidence_text,
        )

    runner = LiveSpecialistRunner(
        analyze,
        lambda _snap, result: done.append(result.secondary_feedback),
        timeout_ms=2000,
    )
    snap = SpecialistSnapshot(
        tenant_id='tenant-smoke',
        meeting_id='meeting-smoke',
        participant_id='remote',
        participant_role='client',
        turn_id='turn-smoke',
        speech_end_ms=int(time.time() * 1000),
        evidence_text='está caro',
        primary_feedback='Pergunte o impacto.',
        conversation_state={'fase_spin': 'problema'},
        host_context='SaaS',
    )
    started = time.perf_counter()
    assert runner.enqueue(snap)
    enqueue_ms = (time.perf_counter() - started) * 1000.0
    assert enqueue_ms < 50.0, f'enqueue blocked: {enqueue_ms:.1f}ms'
    assert done == []
    deadline = time.time() + 2.0
    while not done and time.time() < deadline:
        time.sleep(0.02)
    assert done == ['Smoke secondary tip']
    runner.shutdown()
    _ok(f'specialist enqueue non-blocking | enqueue={enqueue_ms:.1f}ms')


def smoke_secondary_publish_gates() -> None:
    dispatcher = _FakeDispatcher()
    publisher = LiveFeedbackPublisher(
        dispatcher,
        secondary_cooldown_ms=0,
        secondary_max_age_ms=10_000,
        secondary_min_confidence=0.7,
    )
    now = int(time.time() * 1000)
    kwargs = dict(
        meeting_id='meeting-smoke',
        tenant_id='tenant-smoke',
        participant_id='remote',
        participant_role='client',
        parent_turn_id='turn-smoke',
        speech_end_ms=now,
        feedback='Valide a objeção de preço.',
        confidence=0.9,
        feedback_type='objection',
        evidence_text='está caro',
        state={'fase_spin': 'problema'},
        specialist_metadata={'spin': {'fase': 'problema'}},
    )
    assert publisher.publish_secondary_feedback(**kwargs)
    assert publisher.publish_secondary_feedback(**kwargs) is False  # dedupe
    assert len(dispatcher.events) == 1
    event = dispatcher.events[0]
    assert event.metadata['tier'] == 'secondary'
    assert event.metadata['parentTurnId'] == 'turn-smoke'
    assert '_feedbackTier' in event.analysis.conversation_state_json
    _ok('secondary publish + dedupe gates')


def smoke_settings_and_manager_wiring(graphs: LiveTurnGraphs) -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        audio_analysis_mode='live',
        llm_provider='gemini',
        gemini_api_key='AIzaSySmokeTestKeyOnly',
        live_langgraph_enabled=True,
        live_specialist_enabled=False,
    )
    settings.validate()
    assert settings.live_langgraph_enabled is True

    publisher = LiveFeedbackPublisher(_FakeDispatcher())
    manager = GeminiLiveManager(
        api_key='smoke-key',
        publisher=publisher,
        turn_graphs=graphs,
    )
    assert manager._turn_graphs is graphs
    manager.stop()
    _ok('settings validate + GeminiLiveManager wires LiveTurnGraphs')


def run_smoke() -> None:
    graphs = smoke_import_and_compile()
    asyncio.run(smoke_pre_and_post_graphs(graphs))
    smoke_specialist_non_blocking()
    smoke_secondary_publish_gates()
    smoke_settings_and_manager_wiring(graphs)
    print('SMOKE TEST PASSED')


def test_smoke_langgraph_live() -> None:
    """Pytest entrypoint for CI / local `pytest tests/smoke_langgraph_live.py`."""
    run_smoke()


if __name__ == '__main__':
    run_smoke()
