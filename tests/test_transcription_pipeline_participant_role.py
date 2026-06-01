"""Participant role propagation in transcription pipeline events."""

from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.types import TextAnalysisResult, TranscriptionChunk
from src.modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)


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
