"""High-level text-analysis orchestration for transcribed windows."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from ...metrics.realtime_metrics import (
    ANALYSIS_FULL_SIGNAL_TOTAL,
    ANALYSIS_SUPPRESSED_BY_QUALITY_TOTAL,
)
from .sbert_analyzer import SBertAnalyzer
from .semantic_pipeline import SemanticPipeline
from .signals.indecision_signal import IndecisionSignalDetector
from .types import TextAnalysisResult, TranscriptionChunk
from ..transcription.execution_profile import ExecutionProfile


class TextAnalysisService:
    """Analyze transcript text and produce normalized analysis payloads."""

    def __init__(
        self,
        sbert_analyzer: Optional[SBertAnalyzer] = None,
        indecision_detector: Optional[IndecisionSignalDetector] = None,
    ):
        self.sbert_analyzer = sbert_analyzer or SBertAnalyzer()
        self.semantic_pipeline = SemanticPipeline(self.sbert_analyzer)
        self.indecision_detector = indecision_detector or IndecisionSignalDetector()
        self._history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=20))

    def get_analyzer(self) -> SBertAnalyzer:
        """Get the SBERT analyzer."""
        return self.sbert_analyzer

    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded."""
        self.sbert_analyzer.load_sbert_model()

    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """Generate a unique conversational context key."""
        return f"{chunk.meeting_id}:{chunk.participant_id}"

    def analyze(
        self,
        chunk: TranscriptionChunk,
        *,
        execution_profile: ExecutionProfile,
    ) -> TextAnalysisResult:
        """Analyze text and return a normalized analysis result."""
        semantic_result = (
            self.semantic_pipeline.run(
                chunk.text,
                use_embeddings=execution_profile.use_embeddings,
            )
            if execution_profile.semantic_pipeline_enabled
            else {
                'embedding': [],
                'keywords': self.semantic_pipeline.sbert_analyzer.extract_keywords(
                    chunk.text,
                ),
                'sales_category': None,
                'sales_category_confidence': None,
                'category_intensity': None,
                'category_ambiguity': None,
                'category_flags': {},
            }
        )
        context_key = self._get_context_key(chunk)
        history = list(self._history[context_key])

        indecision_metrics = self.indecision_detector.analyze(chunk.text)
        conditional_keywords = self.indecision_detector.detect_conditional_keywords(
            chunk.text,
        )

        category_transition = None
        if execution_profile.compute_category_transition:
            category_transition = self._detect_category_transition(
                semantic_result.get('sales_category'),
                semantic_result.get('sales_category_confidence') or 0.0,
                history,
                chunk.timestamp_ms,
            )

        result = TextAnalysisResult(
            embedding=semantic_result.get('embedding', []),
            keywords=semantic_result.get('keywords', []),
            speech_act=self._infer_speech_act(
                chunk.text,
                conditional_keywords,
                semantic_result.get('sales_category'),
            ),
            sales_category=semantic_result.get('sales_category'),
            sales_category_confidence=semantic_result.get(
                'sales_category_confidence',
            ),
            category_intensity=semantic_result.get('category_intensity'),
            category_ambiguity=semantic_result.get('category_ambiguity'),
            category_flags=semantic_result.get('category_flags', {}),
            conditional_keywords_detected=conditional_keywords,
            indecision_metrics=indecision_metrics,
            category_transition=category_transition,
            analysis_mode=execution_profile.analysis_mode,
            degradation_level=execution_profile.level,
            signal_validity=dict(execution_profile.signal_validity),
            suppression_reasons=list(execution_profile.suppression_reasons),
        )
        if any(not is_valid for is_valid in result.signal_validity.values()):
            ANALYSIS_SUPPRESSED_BY_QUALITY_TOTAL.inc()
        else:
            ANALYSIS_FULL_SIGNAL_TOTAL.inc()

        self._history[context_key].append(
            {
                'sales_category': result.sales_category,
                'timestamp_ms': chunk.timestamp_ms,
                'confidence': result.sales_category_confidence or 0.0,
            },
        )
        return result

    def _infer_speech_act(
        self,
        text: str,
        conditional_keywords: list[str],
        sales_category: Optional[str],
    ) -> str:
        """Infer a coarse speech act for downstream feedback rules."""
        normalized_text = text.strip().lower()
        if normalized_text.endswith('?'):
            return 'question'
        if conditional_keywords:
            return 'conditional_statement'
        if sales_category == 'decision_signal':
            return 'commitment'
        return 'statement'

    def _detect_category_transition(
        self,
        current_category: Optional[str],
        current_confidence: float,
        history: list[dict],
        timestamp_ms: int,
    ) -> Optional[dict]:
        """Detect a meaningful category transition from recent context."""
        if not current_category or not history:
            return None

        last_item = history[-1]
        previous_category = last_item.get('sales_category')
        if not previous_category or previous_category == current_category:
            return None

        return {
            'from_category': previous_category,
            'to_category': current_category,
            'confidence': current_confidence,
            'time_delta_ms': max(0, timestamp_ms - int(last_item.get('timestamp_ms', 0))),
        }