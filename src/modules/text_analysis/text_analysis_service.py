"""High-level text-analysis orchestration for transcribed windows."""

from __future__ import annotations

import json
from typing import Dict, Any, Optional

from .gemini_analyzer import GeminiAnalyzer
from .types import TextAnalysisResult, TranscriptionChunk
from ...config.settings import get_settings


class TextAnalysisService:
    """Analyze transcript text and produce normalized analysis payloads using Gemini."""

    def __init__(
        self,
        gemini_analyzer: Optional[GeminiAnalyzer] = None,
    ):
        settings = get_settings()
        self.gemini_analyzer = gemini_analyzer or GeminiAnalyzer(
            api_key=settings.gemini_api_key or "",
        )
        # Store state per context_key
        self._state: Dict[str, Dict[str, Any]] = {}

    def get_analyzer(self) -> GeminiAnalyzer:
        """Get the Gemini analyzer."""
        return self.gemini_analyzer

    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """Generate a unique conversational context key."""
        return f"{chunk.meeting_id}:{chunk.participant_id}"

    def analyze(
        self,
        chunk: TranscriptionChunk,
    ) -> TextAnalysisResult:
        """Analyze text using Gemini and update conversational state."""
        context_key = self._get_context_key(chunk)
        
        # Get current state or initialize it
        if context_key not in self._state:
            self._state[context_key] = {
                "interesse": "medio",
                "resistencia": "baixa",
                "objecoes_detectadas": [],
                "engajamento": "medio"
            }
            
        current_state = self._state[context_key]
        
        analysis_result = self.gemini_analyzer.analyze(chunk.text, current_state)
        
        # Update our cached state
        self._state[context_key] = analysis_result['conversation_state']
        
        # Return the simplified TextAnalysisResult
        result = TextAnalysisResult(
            direct_feedback=analysis_result['direct_feedback'],
            conversation_state_json=json.dumps(analysis_result['conversation_state'], ensure_ascii=False)
        )

        return result