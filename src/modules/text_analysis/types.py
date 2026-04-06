"""Types used by text analysis and feedback mapping."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptionChunk:
    """Normalized transcription window consumed by text analysis."""

    meeting_id: str
    participant_id: str
    track: str
    text: str
    confidence: float
    timestamp_ms: int
    window_start_ms: int
    window_end_ms: int


@dataclass
class TextAnalysisResult:
    """Normalized text-analysis result for one transcription window."""

    direct_feedback: str = ""
    conversation_state_json: str = "{}"
    confidence: float = 0.5
    feedback_type: Optional[str] = None
    samples_count: Optional[int] = None
    speech_count: Optional[int] = None
    mean_rms_dbfs: Optional[float] = None

    def to_payload_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a payload-friendly dict."""
        payload: Dict[str, Any] = {
            'direct_feedback': self.direct_feedback,
            'conversation_state_json': self.conversation_state_json,
            'confidence': self.confidence,
        }

        if self.feedback_type is not None:
            payload['feedback_type'] = self.feedback_type
        if self.samples_count is not None:
            payload['samples_count'] = self.samples_count
        if self.speech_count is not None:
            payload['speech_count'] = self.speech_count
        if self.mean_rms_dbfs is not None:
            payload['mean_rms_dbfs'] = self.mean_rms_dbfs

        return payload
