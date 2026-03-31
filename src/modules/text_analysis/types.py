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

    embedding: List[float] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    speech_act: Optional[str] = None
    sales_category: Optional[str] = None
    sales_category_confidence: Optional[float] = None
    category_intensity: Optional[float] = None
    category_ambiguity: Optional[float] = None
    category_flags: Dict[str, bool] = field(default_factory=dict)
    conditional_keywords_detected: List[str] = field(default_factory=list)
    indecision_metrics: Optional[Dict[str, Any]] = None
    category_transition: Optional[Dict[str, Any]] = None
    samples_count: Optional[int] = None
    speech_count: Optional[int] = None
    mean_rms_dbfs: Optional[float] = None
    analysis_mode: Optional[str] = None
    degradation_level: Optional[str] = None
    signal_validity: Dict[str, bool] = field(default_factory=dict)
    suppression_reasons: List[str] = field(default_factory=list)

    def to_payload_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a payload-friendly dict."""
        payload: Dict[str, Any] = {
            'embedding': self.embedding,
            'keywords': self.keywords,
            'category_flags': self.category_flags,
            'conditional_keywords_detected': self.conditional_keywords_detected,
        }

        if self.sales_category:
            payload['sales_category'] = self.sales_category
        if self.speech_act:
            payload['speech_act'] = self.speech_act
        if self.sales_category_confidence is not None:
            payload['sales_category_confidence'] = self.sales_category_confidence
        if self.category_intensity is not None:
            payload['category_intensity'] = self.category_intensity
        if self.category_ambiguity is not None:
            payload['category_ambiguity'] = self.category_ambiguity
        if self.indecision_metrics is not None:
            payload['indecision_metrics'] = self.indecision_metrics
        if self.category_transition is not None:
            payload['category_transition'] = self.category_transition
        if self.samples_count is not None:
            payload['samples_count'] = self.samples_count
        if self.speech_count is not None:
            payload['speech_count'] = self.speech_count
        if self.mean_rms_dbfs is not None:
            payload['mean_rms_dbfs'] = self.mean_rms_dbfs
        if self.analysis_mode:
            payload['analysis_mode'] = self.analysis_mode
        if self.degradation_level:
            payload['degradation_level'] = self.degradation_level
        if self.signal_validity:
            payload['signal_validity'] = self.signal_validity
        if self.suppression_reasons:
            payload['suppression_reasons'] = self.suppression_reasons

        return payload
