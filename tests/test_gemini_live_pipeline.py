"""Tests for Gemini Live VAD, cost, turn publish, and settings."""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.config.settings import Settings
from src.modules.audio_buffer.manual_vad import ManualVad
from src.modules.text_analysis.gemini_live_session import GeminiLiveManager
from src.modules.text_analysis.live_cost import MeetingCostTracker, estimate_cost_usd
from src.modules.text_analysis.live_feedback_publisher import LiveFeedbackPublisher
from src.modules.backend_feedback.types import BackendFeedbackEvent


def _pcm_tone(samples: int, amplitude: int = 5000) -> bytes:
    return b''.join(struct.pack('<h', amplitude) for _ in range(samples))


def _pcm_silence(samples: int) -> bytes:
    return b'\x00\x00' * samples


def test_manual_vad_emits_activity_boundaries() -> None:
    vad = ManualVad(sample_rate=16000, channels=1, silence_duration_ms=100, prefix_ms=0)
    speech = _pcm_tone(1600)  # 100ms
    silence = _pcm_silence(1600)

    start_events = vad.push(speech, timestamp_ms=1000)
    kinds = [e.kind for e in start_events]
    assert 'activity_start' in kinds
    assert 'audio' in kinds
    assert vad.speaking is True
    turn_id = start_events[0].turn_id
    assert turn_id

    # Keep speaking
    mid = vad.push(speech, timestamp_ms=1100)
    assert all(e.kind == 'audio' for e in mid)

    # Enough silence to end
    end_events = []
    ts = 1200
    for _ in range(3):
        end_events.extend(vad.push(silence, timestamp_ms=ts))
        ts += 100
    assert any(e.kind == 'activity_end' for e in end_events)
    end = next(e for e in end_events if e.kind == 'activity_end')
    assert end.turn_id == turn_id
    assert end.speech_end_ms is not None
    assert vad.speaking is False


def test_estimate_cost_usd_from_modality_details() -> None:
    usage = {
        'prompt_tokens_details': [
            {'modality': 'AUDIO', 'token_count': 1_000_000},
            {'modality': 'TEXT', 'token_count': 1_000_000},
        ],
        'candidates_tokens_details': [
            {'modality': 'TEXT', 'token_count': 1_000_000},
        ],
    }
    # 3.00 + 0.75 + 4.50 = 8.25
    assert abs(estimate_cost_usd(usage) - 8.25) < 1e-9


def test_meeting_cost_tracker_limits() -> None:
    tracker = MeetingCostTracker(
        meeting_id='m1',
        max_cost_usd=0.01,
        alert_cost_usd=0.005,
    )
    tracker.add_usage(
        {
            'prompt_tokens_details': [
                {'modality': 'AUDIO', 'token_count': 10_000},
            ],
        },
    )
    assert tracker.should_alert() is True
    assert tracker.should_alert() is False  # once
    # Push over limit (10k audio tokens ≈ $0.03)
    tracker.add_usage(
        {
            'prompt_tokens_details': [
                {'modality': 'AUDIO', 'token_count': 10_000},
            ],
        },
    )
    assert tracker.limited is True


class _FakeDispatcher:
    def __init__(self) -> None:
        self.events: list[BackendFeedbackEvent] = []

    def enqueue(self, event: BackendFeedbackEvent) -> bool:
        self.events.append(event)
        return True


def test_live_feedback_publisher_dedupes_turn() -> None:
    dispatcher = _FakeDispatcher()
    publisher = LiveFeedbackPublisher(dispatcher, min_confidence=0.5)
    args = {
        'turnId': 't1',
        'feedback': 'Valide a objeção de preço.',
        'confidence': 0.9,
        'feedback_type': 'objection',
        'evidence_text': 'está caro',
        'estado': {'interesse': 'medio', 'resistencia': 'alta'},
    }
    assert publisher.publish_tool_call(
        meeting_id='meet-1',
        tenant_id='tenant-1',
        participant_id='remote',
        participant_role='client',
        args=args,
        speech_end_ms=1_000,
        turn_id='t1',
    )
    assert publisher.publish_tool_call(
        meeting_id='meet-1',
        tenant_id='tenant-1',
        participant_id='remote',
        participant_role='client',
        args=args,
        speech_end_ms=1_000,
        turn_id='t1',
    ) is False
    assert len(dispatcher.events) == 1
    event = dispatcher.events[0]
    assert event.turn_id == 't1'
    assert event.speech_end_ms == 1_000
    assert event.feedback_trace_id
    assert event.analysis.direct_feedback


def test_live_feedback_publisher_rejects_empty_feedback() -> None:
    dispatcher = _FakeDispatcher()
    publisher = LiveFeedbackPublisher(dispatcher, min_confidence=0.5)
    ok = publisher.publish_tool_call(
        meeting_id='meet-1',
        tenant_id='tenant-1',
        participant_id='remote',
        participant_role='client',
        args={
            'turnId': 't2',
            'feedback': '',
            'confidence': 0.9,
            'feedback_type': 'objection',
            'evidence_text': '',
            'estado': {},
        },
        speech_end_ms=1_000,
        turn_id='t2',
    )
    assert ok is False
    assert dispatcher.events == []


def test_live_mode_settings_validate() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        audio_analysis_mode='live',
        llm_provider='gemini',
        gemini_api_key='AIzaSyTestKey',
        live_silence_duration_ms=250,
        live_max_cost_usd_per_meeting=3.0,
    )
    settings.validate()


def test_live_mode_requires_gemini() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        audio_analysis_mode='live',
        llm_provider='ollama',
    )
    try:
        settings.validate()
    except ValueError as exc:
        assert 'requires LLM_PROVIDER=gemini' in str(exc)
    else:
        raise AssertionError('Expected live mode without gemini to fail')


def test_receive_loop_restarts_after_turn_complete() -> None:
    """SDK receive() ends on turn_complete; we must restart for turn 2+."""

    class _FakeLive:
        def __init__(self) -> None:
            self.calls = 0

        def receive(self):
            self.calls += 1
            call = self.calls

            async def _gen():
                if call == 1:
                    yield SimpleNamespace(
                        usage_metadata=None,
                        session_resumption_update=None,
                        data=None,
                        tool_call=None,
                        server_content=SimpleNamespace(turn_complete=True),
                    )
                    return
                if call == 2:
                    yield SimpleNamespace(
                        usage_metadata=None,
                        session_resumption_update=None,
                        data=None,
                        tool_call=None,
                        server_content=SimpleNamespace(turn_complete=True),
                    )
                    return
                # Keep the loop parked until cancelled on third receive().
                await asyncio.sleep(3600)
                if False:  # pragma: no cover
                    yield None

            return _gen()

    async def _run() -> int:
        manager = GeminiLiveManager(
            api_key='AIzaSyTest',
            publisher=MagicMock(),
        )
        session = SimpleNamespace(
            available=True,
            meeting_id='m1',
            cost=MeetingCostTracker(meeting_id='m1', max_cost_usd=3.0),
            awaiting_tool=True,
            model_turn_done=asyncio.Event(),
            resumption_handle=None,
            send_lock=asyncio.Lock(),
            pending_turn=None,
            tenant_id='t1',
        )
        live = _FakeLive()
        task = asyncio.create_task(
            manager._receive_loop(session, live, types=MagicMock()),
        )
        await asyncio.sleep(0.05)
        assert live.calls >= 2
        assert session.awaiting_tool is False
        assert session.model_turn_done.is_set()
        session.available = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return live.calls

    assert asyncio.run(_run()) >= 2


def test_inject_host_context_does_not_enqueue_realtime() -> None:
    manager = GeminiLiveManager(api_key='AIzaSyTest', publisher=MagicMock())
    from src.modules.text_analysis.gemini_live_session import _MeetingSession

    session = _MeetingSession(
        meeting_id='m1',
        tenant_id='t1',
        api_key='AIzaSyTest',
        model_name='gemini-3.1-flash-live-preview',
        cost=MeetingCostTracker(meeting_id='m1', max_cost_usd=3.0),
    )
    manager._sessions['m1'] = session
    manager.inject_host_context('m1', 'Host said pricing is flexible')
    assert 'pricing' in session.context_summary
    assert session.queue.empty()
