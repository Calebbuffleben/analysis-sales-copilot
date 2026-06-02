"""Participant role propagation in transcription pipeline events."""

from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.types import TextAnalysisResult, TranscriptionChunk
from src.modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)


class _FakeTextAnalysisService:
    def __init__(self):
        self.analyze_calls = []
        self.observe_calls = []

    def analyze(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
        self.analyze_calls.append(chunk)
        return TextAnalysisResult(direct_feedback='Insight', confidence=0.8)

    def observe_context(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
        self.observe_calls.append(chunk)
        return TextAnalysisResult(direct_feedback='', confidence=0.0)


class _FakePublishDispatcher:
    def __init__(self):
        self.events = []

    def enqueue(self, event):
        self.events.append(event)
        return True


def test_build_event_propagates_participant_role() -> None:
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=None,  # type: ignore[arg-type]
        publish_dispatcher=None,  # type: ignore[arg-type]
    )
    transcript = TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='user-abc',
        track='desktop-audio',
        text='Olá, tudo bem?',
        confidence=0.9,
        timestamp_ms=1000,
        window_start_ms=0,
        window_end_ms=1000,
        tenant_id='tenant-1',
        participant_role='host',
    )
    analysis = TextAnalysisResult(direct_feedback='Insight', confidence=0.8)

    event: BackendFeedbackEvent = service._build_event(transcript, analysis)

    assert event.participant_role == 'host'
    assert event.participant_id == 'user-abc'


def test_host_transcript_updates_context_without_publish() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    transcript = TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='user-host',
        track='desktop-audio',
        text='A nossa solução reduz o tempo de follow-up.',
        confidence=0.9,
        timestamp_ms=1000,
        window_start_ms=0,
        window_end_ms=1000,
        tenant_id='tenant-1',
        participant_role='host',
    )

    service.process_transcript('stream-1', transcript)

    assert text_analysis.observe_calls == [transcript]
    assert text_analysis.analyze_calls == []
    assert dispatcher.events == []


def test_participant_transcript_analyzes_and_publishes() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    transcript = TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='user-client',
        track='desktop-audio',
        text='Achei caro comparado ao concorrente.',
        confidence=0.9,
        timestamp_ms=1000,
        window_start_ms=0,
        window_end_ms=1000,
        tenant_id='tenant-1',
        participant_role='participant',
    )

    service.process_transcript('stream-1', transcript)

    assert text_analysis.analyze_calls == [transcript]
    assert text_analysis.observe_calls == []
    assert len(dispatcher.events) == 1
    assert dispatcher.events[0].participant_role == 'participant'


def test_unknown_transcript_is_treated_as_client() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    transcript = TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='legacy-client',
        track='desktop-audio',
        text='Como funciona o próximo passo?',
        confidence=0.9,
        timestamp_ms=1000,
        window_start_ms=0,
        window_end_ms=1000,
        tenant_id='tenant-1',
        participant_role='unknown',
    )

    service.process_transcript('stream-1', transcript)

    assert text_analysis.analyze_calls == [transcript]
    assert dispatcher.events
