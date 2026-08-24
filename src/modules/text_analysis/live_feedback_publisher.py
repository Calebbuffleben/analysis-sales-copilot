"""Publish Gemini Live emit_feedback tool calls with validation and turn dedupe."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Optional, Set

from ...feedback_trace import make_feedback_trace_id
from ...metrics.realtime_metrics import (
    LIVE_SPEECH_END_TO_WS_MS,
    LIVE_TOOL_CALLS_DEDUPED_TOTAL,
    LIVE_TOOL_CALLS_INVALID_TOTAL,
    LIVE_TOOL_CALLS_TOTAL,
    SECONDARY_FEEDBACK_PUBLISHED_TOTAL,
    SECONDARY_FEEDBACK_SUPPRESSED_TOTAL,
    SPEECH_END_TO_SECONDARY_WS_MS,
)
from ...pipeline_latency import LatencyTraceContext, log_speech_to_publish_ms
from ..audio_buffer.prosody_analyzer import ProsodySnapshot
from ..backend_feedback.publish_dispatcher import PublishDispatcher
from ..backend_feedback.types import BackendFeedbackEvent
from .llm_state_validator import (
    build_playbook_hint_json,
    validate_llm_response,
)
from .types import TextAnalysisResult

logger = logging.getLogger(__name__)

PublishHook = Callable[[BackendFeedbackEvent], bool]


class LiveFeedbackPublisher:
    """Validate + dedupe + publish a single final feedback per turnId."""

    def __init__(
        self,
        publish_dispatcher: PublishDispatcher,
        *,
        min_confidence: float = 0.6,
        secondary_enabled: bool = True,
        secondary_min_confidence: float = 0.7,
        secondary_cooldown_ms: int = 15_000,
        secondary_max_age_ms: int = 120_000,
        secondary_types: tuple[str, ...] = ('risk', 'objection'),
        meeting_state: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self._publish_dispatcher = publish_dispatcher
        self._min_confidence = min_confidence
        self._secondary_enabled = secondary_enabled
        self._secondary_min_confidence = secondary_min_confidence
        self._secondary_cooldown_ms = secondary_cooldown_ms
        self._secondary_max_age_ms = secondary_max_age_ms
        self._secondary_types = frozenset(secondary_types)
        self._meeting_state = meeting_state
        self._metrics = metrics
        self._lock = threading.Lock()
        self._seen_turns: Set[str] = set()
        self._secondary_fingerprints: dict[str, int] = {}
        self._last_secondary_ms: dict[str, int] = {}

    def clear_meeting(self, meeting_id: str) -> None:
        prefix = f'{meeting_id}:'
        with self._lock:
            self._seen_turns = {
                key for key in self._seen_turns if not key.startswith(prefix)
            }
            self._secondary_fingerprints = {
                key: value
                for key, value in self._secondary_fingerprints.items()
                if not key.startswith(prefix)
            }
            self._last_secondary_ms.pop(meeting_id, None)

    def observe_turn_metrics(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        args: dict[str, Any],
        prosody: Optional[ProsodySnapshot] = None,
    ) -> None:
        """Feed the live-metrics aggregator; runs even when the turn is dropped.

        Called by the post-tool graph's observe_metrics node (or directly on
        the non-graph fallback path) so monitor snapshots update regardless of
        publish gating (dedupe / empty feedback / low confidence).
        """
        if self._metrics is None:
            return
        try:
            validated = validate_llm_response(args)
            energy = ''
            hesitation = False
            if prosody is not None:
                energy = str(getattr(prosody, 'energy_level', '') or '')
                hesitation = bool(getattr(prosody, 'hesitation_hint', False))
            self._metrics.observe_turn(
                tenant_id=tenant_id,
                meeting_id=meeting_id,
                estado=validated.estado.to_dict(),
                feedback_type=validated.feedback_type,
                confidence=validated.confidence,
                energy_level=energy,
                hesitation_hint=hesitation,
            )
        except Exception:
            logger.debug('live metrics observe failed', exc_info=True)

    def publish_tool_call(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        participant_id: str,
        participant_role: str,
        args: dict[str, Any],
        speech_end_ms: int,
        turn_id: str,
        prosody: Optional[ProsodySnapshot] = None,
    ) -> bool:
        LIVE_TOOL_CALLS_TOTAL.inc()
        dedupe_key = f'{meeting_id}:{turn_id}'
        with self._lock:
            if dedupe_key in self._seen_turns:
                LIVE_TOOL_CALLS_DEDUPED_TOTAL.inc()
                logger.info(
                    'Live turn deduped | meeting=%s | turnId=%s',
                    meeting_id,
                    turn_id,
                )
                return False
            self._seen_turns.add(dedupe_key)

        validated = validate_llm_response(args)
        if self._meeting_state is not None:
            try:
                self._meeting_state.set_conversation(
                    tenant_id,
                    meeting_id,
                    validated.estado.to_dict(),
                )
            except Exception:
                logger.exception('meeting state persist failed')
        feedback = (validated.direct_feedback or '').strip()
        if not feedback:
            LIVE_TOOL_CALLS_INVALID_TOTAL.inc()
            logger.info(
                'Live tool call dropped: empty feedback | meeting=%s | turnId=%s',
                meeting_id,
                turn_id,
            )
            return False
        if validated.confidence < self._min_confidence:
            LIVE_TOOL_CALLS_INVALID_TOTAL.inc()
            logger.info(
                'Live tool call dropped: low confidence | meeting=%s | turnId=%s | conf=%.2f',
                meeting_id,
                turn_id,
                validated.confidence,
            )
            return False

        evidence = (validated.evidence_text or '').strip()
        analysis = TextAnalysisResult(
            direct_feedback=feedback,
            conversation_state_json=json.dumps(
                validated.estado.to_dict(),
                ensure_ascii=False,
            ),
            confidence=validated.confidence,
            feedback_type=validated.feedback_type,
            playbook_hint_json=build_playbook_hint_json(
                validated.playbook_template_key,
                validated.playbook_variables,
            ),
        )
        if prosody is not None:
            analysis.samples_count = prosody.samples_count
            analysis.speech_count = prosody.speech_count
            analysis.mean_rms_dbfs = prosody.mean_rms_dbfs
            analysis.prosody_json = prosody.to_json()
        now_ms = int(time.time() * 1000)
        end_ms = int(speech_end_ms or now_ms)
        trace_id = make_feedback_trace_id(meeting_id, participant_id, end_ms)
        event = BackendFeedbackEvent(
            meeting_id=meeting_id,
            participant_id=participant_id,
            participant_name=None,
            participant_role=participant_role or None,
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=now_ms,
            window_start_ms=max(0, end_ms - 1000),
            window_end_ms=end_ms,
            message='Live emit_feedback ingress event',
            transcript_text=evidence,
            transcript_confidence=validated.confidence,
            analysis=analysis,
            tenant_id=tenant_id,
            turn_id=turn_id,
            speech_end_ms=end_ms,
            feedback_trace_id=trace_id,
        )

        vad_to_now = max(0, now_ms - end_ms)
        LIVE_SPEECH_END_TO_WS_MS.observe(vad_to_now)
        ctx = LatencyTraceContext(
            trace_id=trace_id,
            meeting_id=meeting_id,
            participant_id=participant_id,
            window_end_ms=end_ms,
        )
        log_speech_to_publish_ms(
            logger,
            ctx,
            partial_stable_wall_ms=end_ms,
            transcript_source='live',
        )

        try:
            return bool(self._publish_dispatcher.enqueue(event))
        except Exception:
            logger.exception(
                'Live feedback publish failed | meeting=%s | turnId=%s',
                meeting_id,
                turn_id,
            )
            return False

    def publish_secondary_feedback(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        participant_id: str,
        participant_role: str,
        parent_turn_id: str,
        speech_end_ms: int,
        feedback: str,
        confidence: float,
        feedback_type: str,
        evidence_text: str,
        state: dict[str, Any],
        specialist_metadata: dict[str, Any],
    ) -> bool:
        """Publish a specialist result without touching the primary Live SLO metric."""

        now_ms = int(time.time() * 1000)
        normalized_feedback = ' '.join((feedback or '').split()).strip()
        normalized_type = (feedback_type or '').strip().lower()
        reason = ''
        if not self._secondary_enabled:
            reason = 'disabled'
        elif not normalized_feedback:
            reason = 'empty'
        elif confidence < self._secondary_min_confidence:
            reason = 'low_confidence'
        elif normalized_type not in self._secondary_types:
            reason = 'type'
        elif now_ms - int(speech_end_ms or 0) > self._secondary_max_age_ms:
            reason = 'stale'

        fingerprint = hashlib.sha256(
            f'{meeting_id}|{normalized_type}|{normalized_feedback.lower()}'.encode(),
        ).hexdigest()[:20]
        fingerprint_key = f'{meeting_id}:{fingerprint}'
        with self._lock:
            last_ms = self._last_secondary_ms.get(meeting_id, 0)
            if not reason and now_ms - last_ms < self._secondary_cooldown_ms:
                reason = 'cooldown'
            if not reason and fingerprint_key in self._secondary_fingerprints:
                reason = 'dedupe'
            if reason:
                SECONDARY_FEEDBACK_SUPPRESSED_TOTAL.labels(reason=reason).inc()
                return False
            self._last_secondary_ms[meeting_id] = now_ms
            self._secondary_fingerprints[fingerprint_key] = now_ms
            cutoff = now_ms - self._secondary_max_age_ms
            self._secondary_fingerprints = {
                key: value
                for key, value in self._secondary_fingerprints.items()
                if value >= cutoff
            }

        wire_state = {
            **state,
            '_feedbackTier': 'secondary',
            '_parentTurnId': parent_turn_id,
            '_specialist': specialist_metadata,
        }
        analysis = TextAnalysisResult(
            direct_feedback=normalized_feedback,
            conversation_state_json=json.dumps(wire_state, ensure_ascii=False),
            confidence=confidence,
            feedback_type=normalized_type,
        )
        trace_id = make_feedback_trace_id(
            meeting_id,
            f'{participant_id}:specialist',
            now_ms,
        )
        event = BackendFeedbackEvent(
            meeting_id=meeting_id,
            participant_id=participant_id,
            participant_name=None,
            participant_role=participant_role or None,
            feedback_type='text_analysis_ingress',
            severity='warning' if normalized_type == 'risk' else 'info',
            ts_ms=now_ms,
            window_start_ms=max(0, now_ms - 1),
            window_end_ms=now_ms,
            message='Live specialist secondary feedback',
            transcript_text=(evidence_text or '').strip(),
            transcript_confidence=confidence,
            analysis=analysis,
            tenant_id=tenant_id,
            turn_id=f'{parent_turn_id}:specialist',
            speech_end_ms=speech_end_ms,
            feedback_trace_id=trace_id,
            metadata={
                'tier': 'secondary',
                'parentTurnId': parent_turn_id,
                'specialist': specialist_metadata,
            },
        )
        try:
            accepted = bool(self._publish_dispatcher.enqueue(event))
        except Exception:
            logger.exception(
                'Live specialist publish failed | meeting=%s | turnId=%s',
                meeting_id,
                parent_turn_id,
            )
            return False
        if accepted:
            SECONDARY_FEEDBACK_PUBLISHED_TOTAL.inc()
            SPEECH_END_TO_SECONDARY_WS_MS.observe(
                max(0, now_ms - int(speech_end_ms or now_ms)),
            )
        return accepted
