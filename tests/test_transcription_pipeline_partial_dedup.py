"""Tests for partial/final reconciliation in transcription pipeline."""

from __future__ import annotations

from src.modules.text_analysis.types import TextAnalysisResult, TranscriptionChunk
from src.modules.transcription.partial_turn_coordinator import (
    PartialTurnConfig,
    PartialTurnCoordinator,
)
from src.modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)


class _FakeTextAnalysisService:
    def __init__(self) -> None:
        self.analyze_calls: list[TranscriptionChunk] = []
        self.observe_calls: list[TranscriptionChunk] = []

    def analyze(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
        self.analyze_calls.append(chunk)
        return TextAnalysisResult(direct_feedback='Insight', confidence=0.85)

    def observe_context(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
        self.observe_calls.append(chunk)
        return TextAnalysisResult(direct_feedback='', confidence=0.0)


class _FakePublishDispatcher:
    def __init__(self) -> None:
        self.events: list[object] = []

    def enqueue(self, event: object) -> bool:
        self.events.append(event)
        return True


def _participant_chunk(text: str, *, window_end_ms: int = 5_000) -> TranscriptionChunk:
    return TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='meet-remote',
        track='tab-audio',
        text=text,
        confidence=0.9,
        timestamp_ms=window_end_ms,
        window_start_ms=1_000,
        window_end_ms=window_end_ms,
        tenant_id='tenant-1',
        participant_role='participant',
    )


def test_final_skips_when_partial_already_published_same_text() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    coordinator = PartialTurnCoordinator(PartialTurnConfig(), lambda *_a: None)
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
        partial_coordinator=coordinator,
        partial_min_confidence=0.7,
    )

    text = 'Achei muito caro comparado ao concorrente.'
    service.process_transcript(
        'stream-1',
        _participant_chunk(text, window_end_ms=4_000),
        transcript_source='partial',
    )
    assert len(dispatcher.events) == 1

    service.process_transcript(
        'stream-1',
        _participant_chunk(text, window_end_ms=12_000),
        transcript_source='final',
    )
    assert len(dispatcher.events) == 1
    assert len(text_analysis.analyze_calls) == 1


def test_final_reanalyzes_when_partial_text_diverged() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    coordinator = PartialTurnCoordinator(PartialTurnConfig(), lambda *_a: None)
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
        partial_coordinator=coordinator,
        partial_min_confidence=0.7,
    )

    service.process_transcript(
        'stream-1',
        _participant_chunk('Achei muito caro comparado ao concorrente.', window_end_ms=4_000),
        transcript_source='partial',
    )
    service.process_transcript(
        'stream-1',
        _participant_chunk(
            'Achei muito caro comparado ao concorrente hoje.',
            window_end_ms=12_000,
        ),
        transcript_source='final',
    )
    assert len(text_analysis.analyze_calls) == 2
    assert len(dispatcher.events) == 2


def test_partial_skips_publish_below_confidence() -> None:
    class _LowConfidenceAnalysis(_FakeTextAnalysisService):
        def analyze(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
            self.analyze_calls.append(chunk)
            return TextAnalysisResult(direct_feedback='Weak', confidence=0.5)

    text_analysis = _LowConfidenceAnalysis()
    dispatcher = _FakePublishDispatcher()
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
        partial_min_confidence=0.7,
    )
    service.process_transcript(
        'stream-1',
        _participant_chunk('Achei muito caro comparado ao concorrente.'),
        transcript_source='partial',
    )
    assert dispatcher.events == []


def test_host_publishes_when_flag_enabled() -> None:
    text_analysis = _FakeTextAnalysisService()
    dispatcher = _FakePublishDispatcher()
    service = TranscriptionPipelineService(
        transcription_service=None,  # type: ignore[arg-type]
        text_analysis_service=text_analysis,  # type: ignore[arg-type]
        publish_dispatcher=dispatcher,  # type: ignore[arg-type]
        feedback_allow_host_publish=True,
    )
    host = TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='host',
        track='microphone',
        text='A nossa solução reduz custos.',
        confidence=0.9,
        timestamp_ms=1_000,
        window_start_ms=0,
        window_end_ms=1_000,
        tenant_id='tenant-1',
        participant_role='host',
    )
    service.process_transcript('stream-1', host)
    assert text_analysis.analyze_calls == [host]
    assert len(dispatcher.events) == 1
