"""Publish Gemini Live emit_feedback tool calls with validation and turn dedupe."""

from __future__ import annotations

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
    ) -> None:
        self._publish_dispatcher = publish_dispatcher
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        self._seen_turns: Set[str] = set()

    def clear_meeting(self, meeting_id: str) -> None:
        prefix = f'{meeting_id}:'
        with self._lock:
            self._seen_turns = {
                key for key in self._seen_turns if not key.startswith(prefix)
            }

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
