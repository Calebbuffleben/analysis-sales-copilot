"""Meeting-scoped context behavior for text analysis."""

import json
from collections import deque
from datetime import datetime, timezone

from src.modules.text_analysis.text_analysis_service import (
    _DeferredAnalysis,
    _merge_conversation_state,
    TextAnalysisService,
)
from src.modules.text_analysis.types import TranscriptionChunk


def _chunk(
    *,
    participant_id: str = "participant-1",
    participant_role: str = "participant",
    text: str = "texto de teste",
) -> TranscriptionChunk:
    return TranscriptionChunk(
        meeting_id="meet-1",
        participant_id=participant_id,
        track="desktop-audio",
        text=text,
        confidence=0.9,
        timestamp_ms=1000,
        window_start_ms=0,
        window_end_ms=1000,
        tenant_id="tenant-1",
        participant_role=participant_role,
    )


class _FakeAnalyzer:
    def __init__(self, state_update: dict):
        self.state_update = state_update
        self.calls: list[tuple[str, str]] = []

    def analyze(self, text: str, state: dict, speaker_role: str = "client") -> dict:
        self.calls.append((text, speaker_role))
        return {
            "direct_feedback": "não deve publicar" if speaker_role == "host" else "",
            "confidence": 0.9,
            "feedback_type": "opportunity",
            "conversation_state": self.state_update,
            "playbook_template_key": None,
            "playbook_variables": {},
        }


class _FakeDispatcher:
    def __init__(self):
        self.events = []

    def enqueue(self, event):
        self.events.append(event)
        return True


def _service_with_analyzer(analyzer: _FakeAnalyzer) -> TextAnalysisService:
    service = TextAnalysisService.__new__(TextAnalysisService)
    service.llm_provider = "gemini"
    service.active_analyzer = analyzer
    service._gemini_pool = None
    service._rate_limiter_enabled = False
    service._rate_queue = deque()
    service._rate_queue_lock = None
    service._dispatcher_thread = None
    service._dispatcher_stop = None
    service._publish_dispatcher = None
    service._state = {}
    service._state_metadata = {}
    service._lock = _NullLock()
    service._llm_cache = _NoopCache()
    service._meeting_state = None
    return service


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NoopCache:
    def get(self, text):
        return None

    def put(self, text, result):
        return None


def test_context_key_is_meeting_scoped() -> None:
    service = TextAnalysisService.__new__(TextAnalysisService)

    first = service._get_context_key(_chunk(participant_id="host-1"))
    second = service._get_context_key(_chunk(participant_id="client-1"))

    assert first == "tenant-1:meet-1"
    assert second == "tenant-1:meet-1"


def test_list_merge_dedupes_and_caps() -> None:
    current = {
        "product": "CRM",
        "pain_points": ["follow-up manual"],
        "objections": ["preço alto"],
        "claims": [],
    }
    raw_next = {
        "product": "CR",
        "pain_points": ["Follow-up manual", "dados duplicados"],
        "objections": ["preço alto", "implantação longa"],
        "claims": [f"claim {i}" for i in range(25)],
    }

    merged = _merge_conversation_state(current, raw_next).to_dict()

    assert merged["product"] == "CRM"
    assert merged["pain_points"] == ["follow-up manual", "dados duplicados"]
    assert merged["objections"] == ["preço alto", "implantação longa"]
    assert len(merged["claims"]) == 20


def test_host_turn_enriches_state_without_feedback() -> None:
    analyzer = _FakeAnalyzer(
        {
            "product": "Meet Copilot",
            "pain_points": ["vendedor perde contexto"],
            "claims": ["resume a conversa em tempo real"],
        },
    )
    service = _service_with_analyzer(analyzer)

    result = service.observe_context(
        _chunk(participant_role="host", text="O Meet Copilot resume a conversa."),
    )
    state = json.loads(result.conversation_state_json)

    assert analyzer.calls == [("O Meet Copilot resume a conversa.", "host")]
    assert result.direct_feedback == ""
    assert result.confidence == 0.0
    assert state["product"] == "Meet Copilot"
    assert state["pain_points"] == ["vendedor perde contexto"]
    assert state["claims"] == ["resume a conversa em tempo real"]


def test_deferred_host_analysis_not_published() -> None:
    dispatcher = _FakeDispatcher()
    service = TextAnalysisService.__new__(TextAnalysisService)
    service._publish_dispatcher = dispatcher

    service._publish_deferred_result(
        {
            "direct_feedback": "não publicar",
            "confidence": 0.9,
            "feedback_type": "opportunity",
            "conversation_state": {},
        },
        _chunk(participant_role="host"),
    )

    assert dispatcher.events == []


def test_dispatch_deferred_host_analysis_returns_before_llm() -> None:
    class Slot:
        index = 0
        analyzer = object()

    service = TextAnalysisService.__new__(TextAnalysisService)
    called = False

    def _run_llm_analysis(chunk, current_state, analyzer):
        nonlocal called
        called = True

    service._run_llm_analysis = _run_llm_analysis
    deferred = _DeferredAnalysis(_chunk(participant_role="host"), {})
    service._dispatch_deferred(deferred, Slot())

    assert called is False


def test_get_current_state_initializes_metadata() -> None:
    service = TextAnalysisService.__new__(TextAnalysisService)
    service._state = {}
    service._state_metadata = {}
    service._lock = _NullLock()
    service._meeting_state = None

    state = service._get_current_state("tenant-1:meet-1")

    assert state["product"] == ""
    assert "tenant-1:meet-1" in service._state_metadata
    assert isinstance(
        service._state_metadata["tenant-1:meet-1"]["created_at"],
        datetime,
    )
    assert service._state_metadata["tenant-1:meet-1"]["created_at"].tzinfo == timezone.utc
