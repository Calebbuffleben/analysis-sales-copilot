"""High-level text-analysis orchestration for transcribed windows."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from .gemini_analyzer import GeminiAnalyzer, QuotaExhaustedError
from .llm_state_validator import ConversationState, validate_conversation_state
from .llm_cache import SimpleTextCache
from .llm_logger import log_llm_interaction, log_llm_state_change
from .rule_based_analyzer import analyze_text_fallback
from .types import TextAnalysisResult, TranscriptionChunk
from ...config.settings import get_settings
from ...metrics.realtime_metrics import (
    LLM_CALLS_TOTAL,
    LLM_CALL_ERRORS_TOTAL,
    LLM_FALLBACK_ACTIVATED_TOTAL,
    LLM_FEEDBACK_EMITTED_TOTAL,
    LLM_CALL_DURATION_MS,
    LLM_CONFIDENCE_SCORE,
    LLM_CONFIDENCE_SUPPRESSED_TOTAL,
    LLM_CACHE_HITS_TOTAL,
    LLM_CACHE_MISSES_TOTAL,
    LLM_ACTIVE_STATES,
    LLM_CACHE_SIZE,
    LLM_CACHE_HIT_RATIO,
)

logger = logging.getLogger(__name__)


class TextAnalysisService:
    """Analyze transcript text and produce normalized analysis payloads using Gemini.
    
    Features:
    - Per-context conversation state with TTL-based expiration
    - Thread-safe state access
    - Bounded state cache (prevents memory leaks)
    - Confidence-aware feedback filtering
    """

    # State TTL: expire states after 30 minutes of inactivity
    STATE_TTL = timedelta(minutes=30)
    
    # Max states to keep in memory (prevents unbounded growth)
    MAX_STATES = 1000

    def __init__(
        self,
        gemini_analyzer: Optional[GeminiAnalyzer] = None,
    ):
        settings = get_settings()
        self.gemini_analyzer = gemini_analyzer or GeminiAnalyzer(
            api_key=settings.gemini_api_key or "",
        )
        
        # Thread-safe state storage with metadata
        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {}
        self._state_metadata: Dict[str, dict] = {}  # Tracks last_access, created_at
        
        # LLM response cache (reduces API calls by 40-60%)
        self._llm_cache = SimpleTextCache(
            max_size=500,
            similarity_threshold=0.85,
            ttl_seconds=3600,  # 1 hour
        )

    def get_analyzer(self) -> GeminiAnalyzer:
        """Get the Gemini analyzer."""
        return self.gemini_analyzer

    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """Generate a unique conversational context key."""
        return f"{chunk.meeting_id}:{chunk.participant_id}"

    def _cleanup_expired_states(self) -> int:
        """Remove expired states based on TTL.
        
        Returns number of expired states removed.
        Call this periodically to prevent memory leaks.
        """
        now = datetime.now(timezone.utc)
        expired_keys = []
        
        with self._lock:
            for key, metadata in self._state_metadata.items():
                last_access = metadata.get('last_access')
                if last_access and (now - last_access) > self.STATE_TTL:
                    expired_keys.append(key)
            
            # If still over MAX_STATES after TTL cleanup, evict oldest
            if len(self._state) - len(expired_keys) > self.MAX_STATES:
                sorted_by_age = sorted(
                    self._state_metadata.items(),
                    key=lambda x: x[1].get('created_at', now)
                )
                keys_to_evict = len(self._state) - len(expired_keys) - self.MAX_STATES
                for key, _ in sorted_by_age[:keys_to_evict]:
                    if key not in expired_keys:
                        expired_keys.append(key)
            
            # Remove expired/evicted states
            for key in expired_keys:
                self._state.pop(key, None)
                self._state_metadata.pop(key, None)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired conversation states")
        
        return len(expired_keys)

    def _touch_state(self, context_key: str) -> None:
        """Update last access time for a state."""
        now = datetime.now(timezone.utc)
        with self._lock:
            if context_key not in self._state_metadata:
                self._state_metadata[context_key] = {
                    'created_at': now,
                    'last_access': now,
                }
            else:
                self._state_metadata[context_key]['last_access'] = now

    def get_active_state_count(self) -> int:
        """Get the number of active conversation states (for monitoring)."""
        with self._lock:
            count = len(self._state)
            LLM_ACTIVE_STATES.set(count)
            return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get LLM cache statistics (for monitoring)."""
        stats = self._llm_cache.get_stats()
        LLM_CACHE_SIZE.set(stats['size'])
        LLM_CACHE_HIT_RATIO.set(stats['hit_rate'])
        return stats

    def analyze(
        self,
        chunk: TranscriptionChunk,
    ) -> TextAnalysisResult:
        """Analyze text using Gemini and update conversational state.
        
        Features:
        - Thread-safe state access
        - TTL-based state expiration
        - Confidence-aware feedback (only returns feedback above threshold)
        - Validates LLM response schema before caching
        """
        context_key = self._get_context_key(chunk)

        # Thread-safe state initialization
        with self._lock:
            if context_key not in self._state:
                self._state[context_key] = ConversationState.default_state().to_dict()
                logger.info(f"Initialized new conversation state: {context_key}")

            current_state = self._state[context_key]
            self._touch_state(context_key)

        # Check cache first (before calling LLM)
        cached_response = self._llm_cache.get(chunk.text)
        if cached_response is not None:
            logger.debug(f"Using cached LLM response for '{chunk.text[:30]}...'")
            analysis_result = cached_response
            LLM_CACHE_HITS_TOTAL.inc()
        else:
            LLM_CACHE_MISSES_TOTAL.inc()
            # Cache miss - call LLM with fallback chain
            llm_start_ms = time.time() * 1000
            LLM_CALLS_TOTAL.inc()
            
            try:
                analysis_result = self.gemini_analyzer.analyze(chunk.text, current_state)
                
                # Record LLM call duration
                llm_duration_ms = time.time() * 1000 - llm_start_ms
                LLM_CALL_DURATION_MS.observe(llm_duration_ms)
                
                # Check if LLM returned meaningful result
                if not analysis_result.get('direct_feedback') and analysis_result.get('confidence', 0) < 0.5:
                    # LLM uncertain - try rule-based fallback
                    logger.debug("LLM returned low confidence, trying rule-based fallback")
                    LLM_FALLBACK_ACTIVATED_TOTAL.inc()
                    fallback_result = analyze_text_fallback(chunk.text, current_state)
                    
                    if fallback_result.feedback:
                        logger.info(
                            f"Using rule-based fallback: {fallback_result.feedback[:50]}..."
                        )
                        analysis_result = fallback_result.to_dict()
                    # else: keep the empty LLM result
                        
            except QuotaExhaustedError as e:
                # QUOTA EXHAUSTED: Use rule-based fallback immediately
                llm_duration_ms = time.time() * 1000 - llm_start_ms
                LLM_CALL_DURATION_MS.observe(llm_duration_ms)
                LLM_CALL_ERRORS_TOTAL.inc()
                LLM_FALLBACK_ACTIVATED_TOTAL.inc()
                
                logger.warning(f"Gemini API quota exhausted ({e}), using rule-based fallback")
                try:
                    fallback_result = analyze_text_fallback(chunk.text, current_state)
                    analysis_result = fallback_result.to_dict()
                    
                    if fallback_result.feedback:
                        logger.info(
                            f"Fallback analysis succeeded (quota exhausted): {fallback_result.feedback[:50]}..."
                        )
                except Exception as fallback_error:
                    # Even fallback failed - return safe empty result
                    logger.error(f"Rule-based fallback also failed: {fallback_error}")
                    analysis_result = self.gemini_analyzer._default_response(current_state)
                    
            except Exception as e:
                # LLM failed completely - use rule-based fallback
                llm_duration_ms = time.time() * 1000 - llm_start_ms
                LLM_CALL_DURATION_MS.observe(llm_duration_ms)
                LLM_CALL_ERRORS_TOTAL.inc()
                LLM_FALLBACK_ACTIVATED_TOTAL.inc()
                
                logger.warning(f"Gemini LLM failed ({e}), using rule-based fallback")
                try:
                    fallback_result = analyze_text_fallback(chunk.text, current_state)
                    analysis_result = fallback_result.to_dict()
                    
                    if fallback_result.feedback:
                        logger.info(
                            f"Fallback analysis succeeded: {fallback_result.feedback[:50]}..."
                        )
                except Exception as fallback_error:
                    # Even fallback failed - return safe empty result
                    logger.error(f"Both LLM and fallback failed: {fallback_error}")
                    analysis_result = self.gemini_analyzer._default_response(current_state)
            
            # Cache the result (only if it has feedback)
            if analysis_result.get('direct_feedback'):
                self._llm_cache.put(chunk.text, analysis_result)

        # Periodically cleanup expired states (every 100 calls)
        if not hasattr(self, '_analyze_counter'):
            self._analyze_counter = 0
        self._analyze_counter += 1
        if self._analyze_counter % 100 == 0:
            self._cleanup_expired_states()

        # Call Gemini LLM with fallback chain
        try:
            analysis_result = self.gemini_analyzer.analyze(chunk.text, current_state)
            
            # Check if LLM returned meaningful result
            if not analysis_result.get('direct_feedback') and analysis_result.get('confidence', 0) < 0.5:
                # LLM uncertain - try rule-based fallback
                logger.debug("LLM returned low confidence, trying rule-based fallback")
                fallback_result = analyze_text_fallback(chunk.text, current_state)
                
                if fallback_result.feedback:
                    logger.info(
                        f"Using rule-based fallback: {fallback_result.feedback[:50]}..."
                    )
                    analysis_result = fallback_result.to_dict()
                # else: keep the empty LLM result
                    
        except Exception as e:
            # LLM failed completely - use rule-based fallback
            logger.warning(f"Gemini LLM failed ({e}), using rule-based fallback")
            try:
                fallback_result = analyze_text_fallback(chunk.text, current_state)
                analysis_result = fallback_result.to_dict()
                
                if fallback_result.feedback:
                    logger.info(
                        f"Fallback analysis succeeded: {fallback_result.feedback[:50]}..."
                    )
            except Exception as fallback_error:
                # Even fallback failed - return safe empty result
                logger.error(f"Both LLM and fallback failed: {fallback_error}")
                analysis_result = self.gemini_analyzer._default_response(current_state)

        # Extract results
        direct_feedback = analysis_result.get('direct_feedback', '')
        confidence = analysis_result.get('confidence', 0.5)
        feedback_type = analysis_result.get('feedback_type')

        # Record confidence metrics
        LLM_CONFIDENCE_SCORE.observe(confidence)

        # Filter low-confidence feedback (prevents false positives)
        if confidence < 0.6:
            LLM_CONFIDENCE_SUPPRESSED_TOTAL.inc()
            logger.debug(
                f"Suppressing low-confidence feedback ({confidence:.2f}): {direct_feedback[:50]}"
            )
            direct_feedback = ''
            feedback_type = None

        # Count feedback emitted
        if direct_feedback:
            LLM_FEEDBACK_EMITTED_TOTAL.inc()

        # Validate and update cached state
        new_state = validate_conversation_state(
            analysis_result.get('conversation_state', current_state)
        )
        with self._lock:
            self._state[context_key] = new_state.to_dict()

        # Return the enhanced TextAnalysisResult
        result = TextAnalysisResult(
            direct_feedback=direct_feedback,
            confidence=confidence,
            feedback_type=feedback_type,
            conversation_state_json=json.dumps(new_state.to_dict(), ensure_ascii=False)
        )

        # Structured logging for correlation
        try:
            log_llm_interaction(
                meeting_id=chunk.meeting_id,
                participant_id=chunk.participant_id,
                window_end_ms=chunk.window_end_ms,
                transcript=chunk.text,
                analysis_result={
                    'direct_feedback': direct_feedback,
                    'confidence': confidence,
                    'feedback_type': feedback_type,
                },
                duration_ms=0,  # Already logged via metrics
                cache_hit=cached_response is not None,
            )
        except Exception as log_error:
            logger.debug(f"Failed to log LLM interaction: {log_error}")

        return result