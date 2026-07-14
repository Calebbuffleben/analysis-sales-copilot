"""Thread-safe hub that pushes feedback events to connected desktop clients.

The pipeline threads (STT/analysis/publish) call ``broadcast`` synchronously;
sends are scheduled onto the gateway asyncio loop. The payload mirrors the
backend Socket.IO ``feedback`` event so the desktop normalization is shared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..modules.backend_feedback.types import BackendFeedbackEvent

logger = logging.getLogger(__name__)

RoomKey = tuple[str, str]  # (tenant_id, meeting_id)


class FeedbackHub:
    """Registry of desktop WS connections keyed by (tenantId, meetingId)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rooms: dict[RoomKey, set[Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, tenant_id: str, meeting_id: str, connection: Any) -> None:
        key = (tenant_id, meeting_id)
        with self._lock:
            self._rooms.setdefault(key, set()).add(connection)
        logger.info(
            'ws feedback subscriber joined | tenantId=%s | meetingId=%s | subscribers=%s',
            tenant_id,
            meeting_id,
            len(self._rooms.get(key) or ()),
        )

    def unregister(self, tenant_id: str, meeting_id: str, connection: Any) -> None:
        key = (tenant_id, meeting_id)
        with self._lock:
            room = self._rooms.get(key)
            if room is None:
                return
            room.discard(connection)
            if not room:
                self._rooms.pop(key, None)

    def subscriber_count(self, tenant_id: str, meeting_id: str) -> int:
        with self._lock:
            return len(self._rooms.get((tenant_id, meeting_id)) or ())

    def broadcast(self, event: 'BackendFeedbackEvent') -> bool:
        """Push a feedback event to local subscribers. Returns True if sent.

        Mirrors the backend gate: only events with non-empty
        ``analysis.direct_feedback`` reach the UI.
        """
        direct_feedback = (event.analysis.direct_feedback or '').strip()
        if not direct_feedback:
            return False

        loop = self._loop
        if loop is None or loop.is_closed():
            return False

        key = (event.tenant_id, event.meeting_id)
        with self._lock:
            connections = list(self._rooms.get(key) or ())
        if not connections:
            return False

        started = time.perf_counter()
        payload = self._build_payload(event, direct_feedback)
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.exception('ws feedback payload serialization failed')
            return False

        for connection in connections:
            asyncio.run_coroutine_threadsafe(
                self._safe_send(connection, text),
                loop,
            )
        broadcast_ms = (time.perf_counter() - started) * 1000.0
        speech_end = event.speech_end_ms or event.window_end_ms
        speech_to_ws = max(0, int(time.time() * 1000) - int(speech_end or 0))
        logger.info(
            '⚡ ws feedback broadcast | tenantId=%s | meetingId=%s | subscribers=%s | '
            'feedbackType=%s | turnId=%s | speechEndToWsMs=%s | broadcastSchedMs=%.1f | '
            'traceId=%s',
            event.tenant_id,
            event.meeting_id,
            len(connections),
            event.analysis.feedback_type,
            event.turn_id or '',
            speech_to_ws,
            broadcast_ms,
            event.feedback_trace_id or '',
        )
        return True

    @staticmethod
    async def _safe_send(connection: Any, text: str) -> None:
        try:
            await connection.send(text)
        except Exception:
            # Connection cleanup happens in the gateway handler on close.
            logger.debug('ws feedback send failed (client gone?)', exc_info=True)

    @staticmethod
    def _build_payload(
        event: 'BackendFeedbackEvent',
        direct_feedback: str,
    ) -> dict[str, Any]:
        analysis = event.analysis
        metadata: dict[str, Any] = {
            'source': 'python-direct',
            'transcript': event.transcript_text,
            'transcriptConfidence': event.transcript_confidence,
            'conversationStateJson': analysis.conversation_state_json,
            'confidence': analysis.confidence,
        }
        if analysis.feedback_type:
            metadata['feedbackType'] = analysis.feedback_type
        if (analysis.playbook_hint_json or '').strip():
            metadata['playbookHintJson'] = analysis.playbook_hint_json
        if event.turn_id:
            metadata['turnId'] = event.turn_id
        speech_end = event.speech_end_ms or event.window_end_ms
        if speech_end:
            metadata['speechEndMs'] = int(speech_end)
            metadata['speechAnchorMs'] = int(speech_end)
        if event.feedback_trace_id:
            metadata['feedbackTraceId'] = event.feedback_trace_id

        return {
            'type': 'feedback',
            'payload': {
                'id': f'py-{uuid.uuid4()}',
                'tenantId': event.tenant_id,
                'meetingId': event.meeting_id,
                'participantId': event.participant_id,
                'type': 'llm_insight',
                'severity': event.severity or 'info',
                'ts': int(time.time() * 1000),
                'windowStart': event.window_start_ms,
                'windowEnd': event.window_end_ms,
                'message': direct_feedback,
                'metadata': metadata,
            },
        }
