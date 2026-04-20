"""Types used when publishing canonical feedback events to the backend."""

from dataclasses import dataclass

from ..text_analysis.types import TextAnalysisResult


@dataclass
class BackendFeedbackEvent:
    """Canonical feedback event published to the backend gRPC ingress.

    ``tenant_id`` is mandatory in the multi-tenant backend — it flows through
    as the effective tenant for the SERVICE-role gRPC token and is also
    mirrored in the protobuf payload for defence-in-depth validation.
    """

    meeting_id: str
    participant_id: str
    participant_name: str | None
    participant_role: str | None
    feedback_type: str
    severity: str
    ts_ms: int
    window_start_ms: int
    window_end_ms: int
    message: str
    transcript_text: str
    transcript_confidence: float
    analysis: TextAnalysisResult
    tenant_id: str = ''
