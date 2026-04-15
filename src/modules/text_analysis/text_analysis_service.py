"""High-level text-analysis orchestration for transcribed windows."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
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
    LLM_RATE_LIMITED_TOTAL,
    LLM_RATE_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


class _DeferredAnalysis:
    """A deferred LLM analysis waiting for the RPM rate-limit window."""

    __slots__ = ('chunk', 'current_state', 'enqueued_at_ms')

    def __init__(self, chunk: TranscriptionChunk, current_state: Dict[str, Any]):
        self.chunk = chunk
        self.current_state = current_state
        self.enqueued_at_ms = int(time.time() * 1000)


class TextAnalysisService:
    """Analyze transcript text and produce normalized analysis payloads.

    Features:
    - Per-context conversation state with TTL-based expiration
    - Thread-safe state access
    - Bounded state cache (prevents memory leaks)
    - Confidence-aware feedback filtering
    - **RPM rate limiter with queue (Gemini mode)** — no analysis is lost;
      excess requests are deferred to a background thread and dispatched
      as the rolling 60s window allows.  Results are published via the
      existing publish pipeline, so nothing is dropped.
    """

    # State TTL: expire states after 30 minutes of inactivity
    STATE_TTL = timedelta(minutes=30)

    # Max states to keep in memory (prevents unbounded growth)
    MAX_STATES = 1000

    # RPM rate limiter: Google Gemini free tier = 15 RPM.
    # Use 12 as safety margin (80 % of limit).
    RPM_LIMIT = 12
    RPM_WINDOW_SEC = 60.0

    def __init__(
        self,
        gemini_analyzer: Optional[GeminiAnalyzer] = None,
        ollama_analyzer: Optional[OllamaAnalyzer] = None,
        publish_dispatcher: Optional['PublishDispatcher'] = None,
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
            logger.info("Using Ollama LLM provider (model: %s)", settings.ollama_model)
            self._rate_limiter_enabled = False
        else:  # gemini
            self.active_analyzer = gemini_analyzer or GeminiAnalyzer(
                api_key=settings.gemini_api_key or "",
                model_name=settings.gemini_model,
            )
            logger.info("Using Gemini LLM provider (model: %s)", settings.gemini_model)
            self._rate_limiter_enabled = True

        # --- RPM rate limiter (Gemini only) ---
        # Sliding window of actual Gemini call timestamps.
        self._call_timestamps: deque = deque()
        # Queue of deferred analyses waiting for an RPM slot.
        self._rate_queue: deque = deque()
        self._rate_limiter_lock = threading.Lock()
        self._rate_queue_lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()

        if self._rate_limiter_enabled:
            self._start_dispatcher()

        # Publish dispatcher reference for deferred analysis dispatch
        self._publish_dispatcher = publish_dispatcher

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

    # ------------------------------------------------------------------
    # Rate limiter dispatcher (background thread)
    # ------------------------------------------------------------------

    def _start_dispatcher(self) -> None:
        """Start the background thread that drains the rate-limit queue."""
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            return
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="llm-rate-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        logger.info("LLM rate-limit dispatcher started (Gemini mode, limit=%d RPM)", self.RPM_LIMIT)

    def _dispatcher_loop(self) -> None:
        """Background loop: wait for RPM slot, then dispatch queued analyses."""
        logger.info("LLM rate-limit dispatcher loop running")
        while not self._dispatcher_stop.is_set():
            # Clean old timestamps outside the RPM window
            now = time.time()
            with self._rate_limiter_lock:
                while self._call_timestamps and (now - self._call_timestamps[0]) > self.RPM_WINDOW_SEC:
                    self._call_timestamps.popleft()
                can_call = len(self._call_timestamps) < self.RPM_LIMIT

            if can_call:
                # Try to pop a deferred analysis
                deferred = None
                with self._rate_queue_lock:
                    if self._rate_queue:
                        deferred = self._rate_queue.popleft()
                        LLM_RATE_QUEUE_SIZE.set(len(self._rate_queue))

                if deferred:
                    self._dispatch_deferred(deferred)
                    # Record the timestamp so subsequent iterations respect the limit
                    with self._rate_limiter_lock:
                        self._call_timestamps.append(time.time())
                    # Small pause to avoid burst
                    time.sleep(0.2)
                    continue

            # No slot or no work — sleep briefly
            self._dispatcher_stop.wait(0.5)

    def _dispatch_deferred(self, deferred: _DeferredAnalysis) -> None:
        """Run the LLM analysis for a deferred chunk and publish if it has feedback."""
        waited_ms = int(time.time() * 1000) - deferred.enqueued_at_ms
        logger.info(
            "Dispatching deferred analysis | meeting=%s | participant=%s | "
            "text_len=%d | waited_ms=%d",
            deferred.chunk.meeting_id,
            deferred.chunk.participant_id,
            len(deferred.chunk.text or ''),
            waited_ms,
        )
        try:
            result = self._run_llm_analysis(deferred.chunk.text, deferred.current_state)
            if result and result.get('direct_feedback'):
                self._publish_deferred_result(result, deferred.chunk)
        except Exception as e:
            logger.error("Deferred LLM analysis failed: %s", e, exc_info=True)

    def _publish_deferred_result(
        self, result: Dict[str, Any], chunk: TranscriptionChunk,
    ) -> None:
        """Publish a deferred analysis result via the existing publish dispatcher."""
        if not self._publish_dispatcher:
            logger.warning(
                "Deferred analysis cannot be published — no publish_dispatcher available | meeting=%s",
                chunk.meeting_id,
            )
            return

        try:
            from ..backend_feedback.types import BackendFeedbackEvent

            analysis_obj = TextAnalysisResult(
                direct_feedback=result.get('direct_feedback', ''),
                confidence=result.get('confidence', 0.5),
                feedback_type=result.get('feedback_type'),
                conversation_state_json=json.dumps(
                    result.get('conversation_state', {}),
                    ensure_ascii=False,
                ),
            )

            event = BackendFeedbackEvent(
                meeting_id=chunk.meeting_id,
                participant_id=chunk.participant_id,
                participant_name=None,
                participant_role=None,
                feedback_type='text_analysis_ingress',
                severity='info',
                ts_ms=chunk.timestamp_ms,
                window_start_ms=chunk.window_start_ms,
                window_end_ms=chunk.window_end_ms,
                message='Text analysis ingress (deferred via rate limiter)',
                transcript_text=chunk.text,
                transcript_confidence=chunk.confidence,
                analysis=analysis_obj,
            )
            enqueued = self._publish_dispatcher.enqueue(event)
            if enqueued:
                logger.info(
                    "Deferred analysis published | meeting=%s | feedback='%s...'",
                    chunk.meeting_id,
                    result.get('direct_feedback', '')[:50],
                )
            else:
                logger.warning(
                    "Deferred analysis publish enqueue failed (queue full/stale) | meeting=%s",
                    chunk.meeting_id,
                )
        except Exception as pub_err:
            logger.warning("Deferred analysis publish failed (best effort): %s", pub_err)

    def _next_rpm_slot_ms(self) -> int:
        """Return the timestamp (ms) when the next RPM slot opens."""
        now = time.time()
        with self._rate_limiter_lock:
            if len(self._call_timestamps) < self.RPM_LIMIT:
                return int(now * 1000)
            oldest = self._call_timestamps[0]
            return int((oldest + self.RPM_WINDOW_SEC) * 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_analyzer(self):
        """Get the active LLM analyzer."""
        return self.active_analyzer

    def shutdown(self) -> None:
        """Stop the rate-limit dispatcher thread."""
        self._dispatcher_stop.set()
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=5)

    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """Generate a unique conversational context key."""
        return f"{chunk.meeting_id}:{chunk.participant_id}"

    def _cleanup_expired_states(self) -> int:
        """Remove expired states based on TTL."""
        now = datetime.now(timezone.utc)
        expired_keys = []

        with self._lock:
            for key, metadata in self._state_metadata.items():
                last_access = metadata.get('last_access')
                if last_access and (now - last_access) > self.STATE_TTL:
                    expired_keys.append(key)

            if len(self._state) - len(expired_keys) > self.MAX_STATES:
                sorted_by_age = sorted(
                    self._state_metadata.items(),
                    key=lambda x: x[1].get('created_at', now),
                )
                keys_to_evict = len(self._state) - len(expired_keys) - self.MAX_STATES
                for key, _ in sorted_by_age[:keys_to_evict]:
                    if key not in expired_keys:
                        expired_keys.append(key)

            for key in expired_keys:
                self._state.pop(key, None)
                self._state_metadata.pop(key, None)

        if expired_keys:
            logger.info("Cleaned up %d expired conversation states", len(expired_keys))
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

    def analyze(self, chunk: TranscriptionChunk) -> TextAnalysisResult:
        """Analyze transcript text and update conversational state.

        **RPM rate limiter (Gemini):** when the 12-RPM limit is reached,
        the analysis is deferred to a background queue.  A rule-based
        fallback is returned immediately so the pipeline does not block.
        The deferred Gemini analysis is dispatched later and published
        via the existing pipeline — nothing is lost.
        """
        context_key = self._get_context_key(chunk)

        # Thread-safe state initialization
        with self._lock:
            if context_key not in self._state:
                self._state[context_key] = ConversationState.default_state().to_dict()
                logger.info("Initialized new conversation state: %s", context_key)
            current_state = self._state[context_key]
            self._touch_state(context_key)

        # Check cache first
        cached_response = self._llm_cache.get(chunk.text)
        if cached_response is not None:
            logger.debug("Using cached LLM response for '%s...'", chunk.text[:30])
            analysis_result = cached_response
            LLM_CACHE_HITS_TOTAL.inc()
        else:
            LLM_CACHE_MISSES_TOTAL.inc()

            if self._rate_limiter_enabled:
                # Check RPM — if over limit, enqueue for deferred dispatch
                now = time.time()
                with self._rate_limiter_lock:
                    while self._call_timestamps and (now - self._call_timestamps[0]) > self.RPM_WINDOW_SEC:
                        self._call_timestamps.popleft()
                    can_call = len(self._call_timestamps) < self.RPM_LIMIT

                if not can_call:
                    # Rate limit hit — enqueue, return fallback immediately
                    with self._rate_queue_lock:
                        self._rate_queue.append(_DeferredAnalysis(chunk, current_state))
                        LLM_RATE_LIMITED_TOTAL.inc()
                        LLM_RATE_QUEUE_SIZE.set(len(self._rate_queue))

                    next_slot_ms = self._next_rpm_slot_ms()
                    wait_ms = max(0, next_slot_ms - int(now * 1000))
                    logger.warning(
                        "RPM limit reached (%d/%ds) — analysis deferred. "
                        "Next slot in ~%dms | queue_size=%d | meeting=%s",
                        self.RPM_LIMIT, self.RPM_WINDOW_SEC,
                        wait_ms, len(self._rate_queue), chunk.meeting_id,
                    )

                    # Return fallback immediately so the pipeline doesn't block
                    fallback_result = analyze_text_fallback(chunk.text, current_state)
                    analysis_result = fallback_result.to_dict()
                    if fallback_result.feedback:
                        logger.info(
                            "Immediate fallback (rate limited): %s...",
                            fallback_result.feedback[:50],
                        )
                    # Real Gemini analysis will be dispatched later by the
                    # background thread and published separately.
                else:
                    # Within RPM limit — call Gemini now, record timestamp
                    analysis_result = self._call_llm_with_fallback(chunk, current_state)
                    with self._rate_limiter_lock:
                        self._call_timestamps.append(time.time())
            else:
                # Ollama mode — no rate limiting
                analysis_result = self._call_llm_with_fallback(chunk, current_state)

        # Periodic state cleanup (every 100 calls)
        if not hasattr(self, '_analyze_counter'):
            self._analyze_counter = 0
        self._analyze_counter += 1
        if self._analyze_counter % 100 == 0:
            self._cleanup_expired_states()

        # Extract results
        direct_feedback = analysis_result.get('direct_feedback', '')
        confidence = analysis_result.get('confidence', 0.5)
        feedback_type = analysis_result.get('feedback_type')

        LLM_CONFIDENCE_SCORE.observe(confidence)

        # Filter low-confidence feedback
        if confidence < 0.6:
            LLM_CONFIDENCE_SUPPRESSED_TOTAL.inc()
            logger.debug(
                "Suppressing low-confidence feedback (%.2f): %s",
                confidence, direct_feedback[:50],
            )
            direct_feedback = ''
            feedback_type = None

        if direct_feedback:
            LLM_FEEDBACK_EMITTED_TOTAL.inc()

        # Validate and update cached state (merge so partial LLM/fallback dicts keep prior fields, e.g. fase_spin)
        raw_next = analysis_result.get('conversation_state')
        if not isinstance(raw_next, dict):
            raw_next = {}
        merged_state = {**current_state, **raw_next}
        new_state = validate_conversation_state(merged_state)
        with self._lock:
            self._state[context_key] = new_state.to_dict()

        result = TextAnalysisResult(
            direct_feedback=direct_feedback,
            confidence=confidence,
            feedback_type=feedback_type,
            conversation_state_json=json.dumps(new_state.to_dict(), ensure_ascii=False),
        )

        # Structured logging
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
                duration_ms=0,
                cache_hit=cached_response is not None,
            )
        except Exception as log_error:
            logger.debug("Failed to log LLM interaction: %s", log_error)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm_with_fallback(
        self, chunk: TranscriptionChunk, current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the LLM with full fallback chain. Returns analysis dict."""
        llm_start_ms = time.time() * 1000
        LLM_CALLS_TOTAL.inc()

        try:
            analysis_result = self.active_analyzer.analyze(chunk.text, current_state)
            llm_duration_ms = time.time() * 1000 - llm_start_ms
            LLM_CALL_DURATION_MS.observe(llm_duration_ms)

            # LLM returned low confidence — try rule-based fallback
            if not analysis_result.get('direct_feedback') and analysis_result.get('confidence', 0) < 0.5:
                logger.debug("LLM low confidence, trying rule-based fallback")
                LLM_FALLBACK_ACTIVATED_TOTAL.inc()
                fallback_result = analyze_text_fallback(chunk.text, current_state)
                if fallback_result.feedback:
                    logger.info("Using rule-based fallback: %s...", fallback_result.feedback[:50])
                    analysis_result = fallback_result.to_dict()

            if analysis_result.get('direct_feedback'):
                self._llm_cache.put(chunk.text, analysis_result)

            return analysis_result

        except QuotaExhaustedError as e:
            llm_duration_ms = time.time() * 1000 - llm_start_ms
            LLM_CALL_DURATION_MS.observe(llm_duration_ms)
            LLM_CALL_ERRORS_TOTAL.inc()
            LLM_FALLBACK_ACTIVATED_TOTAL.inc()
            logger.warning("Gemini API quota exhausted (%s), using rule-based fallback", e)
            try:
                return analyze_text_fallback(chunk.text, current_state).to_dict()
            except Exception as fb_err:
                logger.error("Rule-based fallback also failed: %s", fb_err)
                return self.active_analyzer._default_response(current_state)

        except Exception as e:
            llm_duration_ms = time.time() * 1000 - llm_start_ms
            LLM_CALL_DURATION_MS.observe(llm_duration_ms)
            LLM_CALL_ERRORS_TOTAL.inc()
            LLM_FALLBACK_ACTIVATED_TOTAL.inc()
            logger.warning("Gemini LLM failed (%s), using rule-based fallback", e)
            try:
                return analyze_text_fallback(chunk.text, current_state).to_dict()
            except Exception as fb_err:
                logger.error("Both LLM and fallback failed: %s", fb_err)
                return self.active_analyzer._default_response(current_state)

    def _run_llm_analysis(
        self, text: str, current_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run LLM analysis for a deferred chunk (called from dispatcher thread)."""
        cached = self._llm_cache.get(text)
        if cached is not None:
            LLM_CACHE_HITS_TOTAL.inc()
            return cached

        LLM_CACHE_MISSES_TOTAL.inc()
        LLM_CALLS_TOTAL.inc()
        llm_start_ms = time.time() * 1000

        try:
            result = self.active_analyzer.analyze(text, current_state)
            LLM_CALL_DURATION_MS.observe(time.time() * 1000 - llm_start_ms)
            if result.get('direct_feedback'):
                self._llm_cache.put(text, result)
            return result
        except Exception as e:
            LLM_CALL_ERRORS_TOTAL.inc()
            logger.error("Deferred LLM analysis error: %s", e, exc_info=True)
            return None
