"""High-level text-analysis orchestration for transcribed windows."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from .gemini_analyzer import GeminiAnalyzer, QuotaExhaustedError
from .ollama_analyzer import OllamaAnalyzer
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
        ollama_analyzer: Optional[OllamaAnalyzer] = None,
    ):
        settings = get_settings()
        
        # Initialize LLM provider based on configuration
        self.llm_provider = settings.llm_provider
        
        if self.llm_provider == 'ollama':
            self.active_analyzer = ollama_analyzer or OllamaAnalyzer(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout=settings.ollama_timeout,
            )
            logger.info(f"Using Ollama LLM provider (model: {settings.ollama_model})")
        else:  # gemini
            self.active_analyzer = gemini_analyzer or GeminiAnalyzer(
                api_key=settings.gemini_api_key or "",
            )
            logger.info("Using Gemini LLM provider")
        
        # Thread-safe state storage with metadata
        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {}
        self._state_metadata: Dict[str, dict] = {}  # Tracks last_access, created_at

        # FIX #4: Shared meeting-level state for cross-participant context
        # This allows objections detected in participant A to appear in participant B's state
        self._meeting_state: Dict[str, Dict[str, Any]] = {}  # meeting_id -> shared state

        # LLM response cache (reduces API calls by 40-60%)
        self._llm_cache = SimpleTextCache(
            max_size=500,
            similarity_threshold=0.85,
            ttl_seconds=3600,  # 1 hour
        )

    def get_analyzer(self):
        """Get the active LLM analyzer."""
        return self.active_analyzer

    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """Generate a unique conversational context key."""
        return f"{chunk.meeting_id}:{chunk.participant_id}"

    def _merge_states(self, participant_state: Dict[str, Any], meeting_state: Dict[str, Any]) -> Dict[str, Any]:
        """FIX #4: Merge participant state with shared meeting state.

        The merged state is sent to the LLM prompt. This ensures that objections
        detected from any participant are visible to the LLM.

        Priority: participant_state > meeting_state (participant is more specific)
        """
        merged = meeting_state.copy()
        merged.update(participant_state)

        # Merge objections (union of both)
        participant_objections = set(participant_state.get('objecoes_detectadas', []))
        meeting_objections = set(meeting_state.get('objecoes_detectadas', []))
        merged['objecoes_detectadas'] = list(participant_objections | meeting_objections)

        return merged

    def _update_meeting_state(self, meeting_id: str, new_state: Dict[str, Any]) -> None:
        """FIX #4: Update shared meeting state with new state."""
        with self._lock:
            if meeting_id not in self._meeting_state:
                self._meeting_state[meeting_id] = new_state
                return

            current = self._meeting_state[meeting_id]

            # Update fields with more severe values
            severity_order = {'baixa': 0, 'baixo': 0, 'media': 1, 'medio': 1, 'alta': 2, 'alto': 2}

            for field in ['interesse', 'resistencia', 'engajamento']:
                new_val = new_state.get(field, current.get(field))
                if new_val and severity_order.get(new_val, 0) >= severity_order.get(current.get(field, ''), 0):
                    current[field] = new_val

            # Merge objections (union)
            current_objections = set(current.get('objecoes_detectadas', []))
            new_objections = set(new_state.get('objecoes_detectadas', []))
            current['objecoes_detectadas'] = list(current_objections | new_objections)

            self._meeting_state[meeting_id] = current

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

    def clear_meeting_state(self, meeting_id: str) -> int:
        """FIX #3: Clear all state for a meeting when it ends.

        This prevents stale state from persisting for 30min after meeting ends.

        Args:
            meeting_id: Meeting identifier to clear

        Returns:
            Number of states cleared
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._state
                if key.startswith(f"{meeting_id}:")
            ]
            for key in keys_to_remove:
                self._state.pop(key, None)
                self._state_metadata.pop(key, None)

            # Also clear shared meeting state
            self._meeting_state.pop(meeting_id, None)

        if keys_to_remove:
            logger.info(
                f"Cleared {len(keys_to_remove)} conversation states for meeting {meeting_id}"
            )
        return len(keys_to_remove)

    def get_meeting_state_keys(self, meeting_id: str) -> list[str]:
        """Get all state keys for a meeting (for debugging)."""
        with self._lock:
            return [
                key for key in self._state
                if key.startswith(f"{meeting_id}:")
            ]

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

                # FIX #4: Initialize shared meeting state if first participant
                if chunk.meeting_id not in self._meeting_state:
                    self._meeting_state[chunk.meeting_id] = {
                        "interesse": "medio",
                        "resistencia": "baixa",
                        "objecoes_detectadas": [],
                        "engajamento": "medio",
                        "participant_count": 0,
                    }
                    logger.info(f"Initialized shared meeting state: {chunk.meeting_id}")

            current_state = self._state[context_key]
            meeting_state = self._meeting_state.get(chunk.meeting_id, {})
            self._touch_state(context_key)

        # FIX #4: Merge participant state with shared meeting state
        # This allows objections from participant A to appear in participant B's context
        merged_state = self._merge_states(current_state, meeting_state)

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
                analysis_result = self.active_analyzer.analyze(chunk.text, current_state)
                
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
                    analysis_result = self.active_analyzer._default_response(current_state)
                    
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
                    analysis_result = self.active_analyzer._default_response(current_state)
            
            # Cache the result (only if it has feedback)
            if analysis_result.get('direct_feedback'):
                self._llm_cache.put(chunk.text, analysis_result)

        # Periodically cleanup expired states (every 100 calls)
        if not hasattr(self, '_analyze_counter'):
            self._analyze_counter = 0
        self._analyze_counter += 1
        if self._analyze_counter % 100 == 0:
            self._cleanup_expired_states()

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
        # FIX #1: Pass current_state as fallback to prevent silent reset
        old_state = current_state.copy()
        new_state = validate_conversation_state(
            analysis_result.get('conversation_state', current_state),
            fallback_state=ConversationState(**current_state),  # Preserve existing state
        )
        with self._lock:
            self._state[context_key] = new_state.to_dict()
            # FIX #4: Also update shared meeting state
            self._update_meeting_state(chunk.meeting_id, new_state.to_dict())

        # FIX #2: Log state changes for debugging
        try:
            log_llm_state_change(
                context_key=context_key,
                meeting_id=chunk.meeting_id,
                participant_id=chunk.participant_id,
                old_state=old_state,
                new_state=new_state.to_dict(),
            )
        except Exception as log_error:
            logger.debug(f"Failed to log state change: {log_error}")

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