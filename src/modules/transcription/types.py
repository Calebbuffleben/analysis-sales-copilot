"""Types used by the transcription pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    """Represents the transcription output for one ready window."""

    text: str
    confidence: float
    language: Optional[str] = None
    segment_count: int = 0
    vad_filter_used: bool = True
    empty_reason: Optional[str] = None
    diagnostic_no_vad_chars: int = 0
