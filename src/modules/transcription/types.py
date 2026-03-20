"""Types used by the transcription pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    """Represents the transcription output for one ready window."""

    text: str
    confidence: float
    language: Optional[str] = None
