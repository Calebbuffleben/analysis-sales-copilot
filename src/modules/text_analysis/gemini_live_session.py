"""Gemini Live session manager for sub-second final feedback.

Architecture (ponytail):
- One asyncio loop in a background thread.
- One Live session per meeting for the client (tab-audio) stream.
- Manual VAD delimits turns; Gemini alone emits feedback via emit_feedback.
- generateContent multimodal remains the degraded fallback outside the 1s budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...metrics.realtime_metrics import (
    LIVE_AUDIO_BYTES_SENT_TOTAL,
    LIVE_COST_LIMIT_TRIPS_TOTAL,
    LIVE_COST_USD_PER_MEETING,
    LIVE_COST_USD_TOTAL,
    LIVE_FALLBACK_TOTAL,
    LIVE_SESSIONS_CLOSED_TOTAL,
    LIVE_SESSIONS_OPEN,
    LIVE_SESSIONS_RESUMED_TOTAL,
    LIVE_SESSIONS_STARTED_TOTAL,
    LIVE_UNEXPECTED_AUDIO_BYTES_TOTAL,
    LIVE_VAD_END_TO_TOOL_CALL_MS,
)
from ..audio_buffer.manual_vad import ManualVad, VadEvent
from ..audio_buffer.prosody_analyzer import (
    TURN_PCM_MAX_BYTES,
    ProsodySnapshot,
    analyze_turn_prosody,
)
from ..audio_buffer.service import WAV_HEADER_BYTES
from .gemini_transport import uses_vertex_express_api_key
from .live_cost import MeetingCostTracker
from .live_feedback_publisher import LiveFeedbackPublisher
from ..playbooks.catalog_cache import PlaybookCatalogCache
from ..playbooks.resolve import format_catalog_for_prompt
from ..playbooks.retrieve import (
    CATALOG_PROMPT_MAX,
    RETRIEVE_MIN_TEMPLATES,
    PlaybookIndex,
    build_retrieve_query,
    format_retrieve_nudge,
    hint_from_emit_feedback_args,
)

logger = logging.getLogger(__name__)

# Fail-open: never block Live publish waiting for prosody.
_PROSODY_PUBLISH_WAIT_S = 0.03
_PROSODY_CACHE_MAX = 8

EMIT_FEEDBACK_TOOL = {
    'function_declarations': [
        {
            'name': 'emit_feedback',
            'description': (
                'Emit exactly one final coaching feedback for the just-finished '
                'customer speech turn. Call once per turnId; each new turnId needs a '
                'new call. Never reuse an old turnId. Never speak aloud — only call '
                'this tool.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'turnId': {
                        'type': 'string',
                        'description': 'Opaque turn id provided by the server for this utterance.',
                    },
                    'feedback': {
                        'type': 'string',
                        'description': 'Short actionable coaching tip for the seller, or empty if none.',
                    },
                    'confidence': {
                        'type': 'number',
                        'description': 'Confidence 0..1',
                    },
                    'feedback_type': {
                        'type': 'string',
                        'description': (
                            'objection|opportunity|rapport|closing|clarification|risk|null'
                        ),
                    },
                    'evidence_text': {
                        'type': 'string',
                        'description': 'Short literal quote from the customer speech.',
                    },
                    'estado': {
                        'type': 'object',
                        'description': 'Partial conversation state delta.',
                    },
                    'playbook_template_key': {
                        'type': 'string',
                    },
                    'playbook_variables': {
                        'type': 'object',
                    },
                },
                'required': [
                    'turnId',
                    'feedback',
                    'confidence',
                    'feedback_type',
                    'evidence_text',
                    'estado',
                ],
            },
        },
    ],
}

SYSTEM_INSTRUCTION = (
    'Você é um copiloto de vendas de baixa latência. Ouça o áudio do CLIENTE. '
    'Sempre que um turno de fala do cliente terminar, chame emit_feedback exatamente '
    'uma vez com o turnId daquele turno. Cada novo turnId é um turno novo — chame de '
    'novo. Nunca reutilize um turnId antigo. Nunca responda por voz. Se não houver '
    'feedback útil, chame emit_feedback com feedback="" e confidence=0. Priorize '
    'objection, opportunity, rapport, closing, clarification e risk.'
)


def _extract_pcm(wav_or_pcm: bytes) -> bytes:
    if len(wav_or_pcm) >= WAV_HEADER_BYTES and wav_or_pcm[:4] == b'RIFF':
        return wav_or_pcm[WAV_HEADER_BYTES:]
    return wav_or_pcm


def _is_host_role(role: object) -> bool:
    return str(role or '').strip().lower() == 'host'


@dataclass
class _PendingTurn:
    turn_id: str
    speech_end_ms: int
    meeting_id: str
    tenant_id: str
    participant_id: str
    participant_role: str


@dataclass
class _MeetingSession:
    meeting_id: str
    tenant_id: str
    api_key: str
    model_name: str
    cost: MeetingCostTracker
    vad: ManualVad = field(default_factory=ManualVad)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    available: bool = True
    resumption_handle: Optional[str] = None
    pending_turn: Optional[_PendingTurn] = None
    context_summary: str = ''
    opened_wall_ms: int = 0
    rotation_minutes: float = 2.0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # True after activity_end until tool call handled or timeout.
    awaiting_tool: bool = False
    model_turn_done: asyncio.Event = field(default_factory=asyncio.Event)
    catalog_prompt: str = ''
    playbook_index: Optional[PlaybookIndex] = None
    retrieve_query_hint: str = ''
    # Prosody enrichment (parallel to Live; never blocks send path).
    sample_rate: int = 16_000
    channels: int = 1
    turn_pcm: bytearray = field(default_factory=bytearray)
    prosody_by_turn: dict[str, ProsodySnapshot] = field(default_factory=dict)
    last_distinctive_prosody: Optional[ProsodySnapshot] = None
    _prosody_tasks: dict[str, asyncio.Task] = field(default_factory=dict)


class GeminiLiveManager:
    """Owns Live sessions and routes client PCM through manual VAD."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = 'gemini-3.1-flash-live-preview',
        publisher: LiveFeedbackPublisher,
        max_cost_usd_per_meeting: float = 3.0,
        alert_cost_usd: float = 1.0,
        max_concurrent_sessions: int = 20,
        silence_duration_ms: int = 250,
        min_speech_ms: int = 400,
        context_window_tokens: int = 12_000,
        session_rotation_minutes: float = 2.0,
        on_unavailable: Optional[Callable[[str], None]] = None,
        catalog_cache: Optional[PlaybookCatalogCache] = None,
    ) -> None:
        self._api_key = (api_key or '').strip()
        self._model_name = model_name
        self._publisher = publisher
        self._max_cost = max_cost_usd_per_meeting
        self._alert_cost = alert_cost_usd
        self._max_sessions = max_concurrent_sessions
        self._silence_ms = silence_duration_ms
        self._min_speech_ms = min_speech_ms
        self._context_window_tokens = context_window_tokens
        self._rotation_minutes = session_rotation_minutes
        self._on_unavailable = on_unavailable
        self._catalog_cache = catalog_cache

        self._lock = threading.Lock()
        self._sessions: dict[str, _MeetingSession] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name='gemini-live-manager',
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown_all(), loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def is_available(self, meeting_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(meeting_id)
            return bool(session and session.available and not session.cost.limited)

    def mark_unavailable(self, meeting_id: str, reason: str) -> None:
        with self._lock:
            session = self._sessions.get(meeting_id)
            if session is None:
                return
            session.available = False
        LIVE_FALLBACK_TOTAL.inc()
        logger.warning(
            'Live session unavailable | meeting=%s | reason=%s',
            meeting_id,
            reason,
        )
        if self._on_unavailable:
            try:
                self._on_unavailable(meeting_id)
            except Exception:
                logger.exception('on_unavailable callback failed')

    def ensure_session(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bool:
        if not self._api_key:
            return False
        with self._lock:
            existing = self._sessions.get(meeting_id)
            if existing is not None:
                return existing.available and not existing.cost.limited
            if len(self._sessions) >= self._max_sessions:
                logger.warning(
                    'Live max concurrent sessions reached | max=%s',
                    self._max_sessions,
                )
                return False
            session = _MeetingSession(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                api_key=self._api_key,
                model_name=self._model_name,
                cost=MeetingCostTracker(
                    meeting_id=meeting_id,
                    max_cost_usd=self._max_cost,
                    alert_cost_usd=self._alert_cost,
                ),
                vad=ManualVad(
                    sample_rate=sample_rate,
                    channels=channels,
                    silence_duration_ms=self._silence_ms,
                    min_speech_ms=self._min_speech_ms,
                ),
                rotation_minutes=self._rotation_minutes,
                opened_wall_ms=int(time.time() * 1000),
                sample_rate=sample_rate,
                channels=channels,
            )
            self._sessions[meeting_id] = session

        # Warm playbook catalog off the audio hot path (once per meeting).
        if self._catalog_cache is not None and tenant_id:
            try:
                templates = self._catalog_cache.get(tenant_id)
                n = len(templates)
                session.playbook_index = None
                session.retrieve_query_hint = ''
                if n > RETRIEVE_MIN_TEMPLATES:
                    index = PlaybookIndex.from_templates(templates)
                    session.playbook_index = index
                    if n > CATALOG_PROMPT_MAX:
                        ranked = index.top_templates_for_prompt(
                            session.context_summary,
                            max_items=CATALOG_PROMPT_MAX,
                        )
                        session.catalog_prompt = format_catalog_for_prompt(ranked)
                    else:
                        session.catalog_prompt = format_catalog_for_prompt(templates)
                else:
                    session.catalog_prompt = format_catalog_for_prompt(templates)
                logger.info(
                    'playbook.catalog_loaded | tenant=%s | meeting=%s | n=%s | '
                    'retrieve=%s',
                    tenant_id,
                    meeting_id,
                    n,
                    bool(session.playbook_index),
                )
            except Exception:
                logger.exception(
                    'playbook.catalog_warm_failed | tenant=%s | meeting=%s',
                    tenant_id,
                    meeting_id,
                )

        loop = self._wait_for_loop()
        if loop is None:
            return False
        fut = asyncio.run_coroutine_threadsafe(self._start_session(session), loop)
        try:
            fut.result(timeout=15.0)
            return True
        except Exception:
            logger.exception('Failed to start Live session | meeting=%s', meeting_id)
            self.mark_unavailable(meeting_id, 'start_failed')
            return False

    def push_audio(
        self,
        *,
        meeting_id: str,
        tenant_id: str,
        participant_id: str,
        participant_role: str,
        track: str,
        wav_or_pcm: bytes,
        timestamp_ms: int,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bool:
        """Route client audio into Live. Returns False if Live unavailable."""
        if _is_host_role(participant_role):
            return False
        if not self.ensure_session(
            meeting_id=meeting_id,
            tenant_id=tenant_id,
            sample_rate=sample_rate,
            channels=channels,
        ):
            return False

        with self._lock:
            session = self._sessions.get(meeting_id)
            if session is None or not session.available or session.cost.limited:
                return False
            if session.vad._sample_rate != sample_rate or session.vad._channels != channels:
                session.vad = ManualVad(
                    sample_rate=sample_rate,
                    channels=channels,
                    silence_duration_ms=self._silence_ms,
                    min_speech_ms=self._min_speech_ms,
                )
                session.sample_rate = sample_rate
                session.channels = channels

        pcm = _extract_pcm(wav_or_pcm)
        if not pcm:
            return True

        events = session.vad.push(pcm, timestamp_ms)
        loop = self._loop
        if loop is None:
            return False
        for event in events:
            asyncio.run_coroutine_threadsafe(
                session.queue.put(
                    (
                        event,
                        {
                            'meeting_id': meeting_id,
                            'tenant_id': tenant_id,
                            'participant_id': participant_id,
                            'participant_role': participant_role,
                            'track': track,
                        },
                    ),
                ),
                loop,
            )
        return True

    def end_meeting(self, meeting_id: str) -> None:
        loop = self._loop
        with self._lock:
            session = self._sessions.pop(meeting_id, None)
        if session is None:
            return
        self._publisher.clear_meeting(meeting_id)
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self._close_session(session), loop)

    def inject_host_context(self, meeting_id: str, summary: str) -> None:
        """Store host summary for later session seed — never send mid-call.

        Sending host text via realtime_input mid-session starts a model turn on
        Gemini 3.1 Live and blocks subsequent client emit_feedback tool calls.
        """
        text = (summary or '').strip()
        if not text:
            return
        with self._lock:
            session = self._sessions.get(meeting_id)
            if session is None:
                return
            session.context_summary = text[:1500]

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            while not self._stop.is_set():
                loop.run_until_complete(asyncio.sleep(0.2))
        finally:
            try:
                loop.run_until_complete(self._shutdown_all())
            except Exception:
                pass
            loop.close()
            self._loop = None

    def _wait_for_loop(self, timeout_sec: float = 5.0) -> Optional[asyncio.AbstractEventLoop]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._loop is not None:
                return self._loop
            time.sleep(0.05)
        return self._loop

    async def _shutdown_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await self._close_session(session)

    async def _start_session(self, session: _MeetingSession) -> None:
        if session.task and not session.task.done():
            return
        session.task = asyncio.create_task(self._session_loop(session))

    async def _close_session(self, session: _MeetingSession) -> None:
        session.available = False
        try:
            await session.queue.put(None)
        except Exception:
            pass
        if session.task:
            try:
                await asyncio.wait_for(session.task, timeout=5.0)
            except Exception:
                session.task.cancel()
        LIVE_SESSIONS_CLOSED_TOTAL.inc()

    async def _session_loop(self, session: _MeetingSession) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=session.api_key,
            vertexai=uses_vertex_express_api_key(session.api_key),
        )
        LIVE_SESSIONS_STARTED_TOTAL.inc()
        LIVE_SESSIONS_OPEN.inc()
        session.opened_wall_ms = int(time.time() * 1000)

        config = self._build_config(types, session)
        try:
            async with client.aio.live.connect(
                model=session.model_name,
                config=config,
            ) as live:
                if session.resumption_handle:
                    LIVE_SESSIONS_RESUMED_TOTAL.inc()
                recv_task = asyncio.create_task(self._receive_loop(session, live, types))
                try:
                    while session.available:
                        item = await session.queue.get()
                        if item is None:
                            break
                        # Legacy queue item — host context must not hit realtime mid-call.
                        if isinstance(item, tuple) and item and item[0] == 'host_context':
                            continue
                        event, meta = item
                        assert isinstance(event, VadEvent)
                        await self._handle_vad_event(session, live, types, event, meta)
                        if session.cost.limited:
                            LIVE_COST_LIMIT_TRIPS_TOTAL.inc()
                            self.mark_unavailable(session.meeting_id, 'cost_limit')
                            break
                        if self._should_rotate(session):
                            # Rotate between turns only.
                            if not session.vad.speaking and not session.awaiting_tool:
                                session.resumption_handle = None
                                break
                finally:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except Exception:
                        pass
        except Exception as exc:
            logger.exception(
                'Live session crashed | meeting=%s | error=%s',
                session.meeting_id,
                exc,
            )
            self.mark_unavailable(session.meeting_id, f'crash:{type(exc).__name__}')
        finally:
            LIVE_SESSIONS_OPEN.dec()

        # Soft rotate: reopen if still marked available and under cost.
        with self._lock:
            still = self._sessions.get(session.meeting_id)
        if still is session and session.available and not session.cost.limited:
            session.opened_wall_ms = int(time.time() * 1000)
            session.task = asyncio.create_task(self._session_loop(session))

    def _build_config(self, types: Any, session: _MeetingSession) -> Any:
        thinking = None
        try:
            thinking = types.ThinkingConfig(thinking_level='minimal')
        except Exception:
            thinking = None

        system_instruction = SYSTEM_INSTRUCTION
        if session.catalog_prompt:
            system_instruction = f'{SYSTEM_INSTRUCTION}\n\n{session.catalog_prompt}'

        kwargs: dict[str, Any] = {
            'response_modalities': ['AUDIO'],
            'system_instruction': system_instruction,
            'tools': [EMIT_FEEDBACK_TOOL],
            'realtime_input_config': {
                'automatic_activity_detection': {'disabled': True},
            },
            'context_window_compression': {
                'sliding_window': {},
                'trigger_tokens': self._context_window_tokens,
            },
        }
        if thinking is not None:
            kwargs['thinking_config'] = thinking
        if session.resumption_handle:
            kwargs['session_resumption'] = {'handle': session.resumption_handle}
        else:
            kwargs['session_resumption'] = {}
        try:
            return types.LiveConnectConfig(**kwargs)
        except Exception:
            # Older SDK shapes: plain dict is accepted by connect().
            return kwargs

    async def _handle_vad_event(
        self,
        session: _MeetingSession,
        live: Any,
        types: Any,
        event: VadEvent,
        meta: dict[str, str],
    ) -> None:
        if event.kind == 'activity_start':
            session.turn_pcm.clear()
            logger.info(
                'live.vad.start | meeting=%s | turnId=%s',
                session.meeting_id,
                event.turn_id,
            )
            async with session.send_lock:
                await live.send_realtime_input(activity_start=types.ActivityStart())
                await live.send_realtime_input(
                    text=(
                        f'Turno iniciado. turnId="{event.turn_id}". '
                        'Ao final deste turno chame emit_feedback uma única vez '
                        'com este turnId.'
                    ),
                )
            return

        if event.kind == 'audio' and event.pcm:
            self._append_turn_pcm(session, event.pcm)
            async with session.send_lock:
                await live.send_realtime_input(
                    audio=types.Blob(
                        data=event.pcm,
                        mime_type='audio/pcm;rate=16000',
                    ),
                )
            LIVE_AUDIO_BYTES_SENT_TOTAL.inc(len(event.pcm))
            return

        if event.kind == 'activity_end':
            session.pending_turn = _PendingTurn(
                turn_id=event.turn_id,
                speech_end_ms=int(event.speech_end_ms or time.time() * 1000),
                meeting_id=meta['meeting_id'],
                tenant_id=meta['tenant_id'],
                participant_id=meta['participant_id'],
                participant_role=meta['participant_role'],
            )
            session.awaiting_tool = True
            session.model_turn_done.clear()
            turn_pcm = bytes(session.turn_pcm)
            session.turn_pcm.clear()
            self._schedule_prosody(session, event.turn_id, turn_pcm)
            logger.info(
                'live.vad.end | meeting=%s | turnId=%s',
                session.meeting_id,
                event.turn_id,
            )
            async with session.send_lock:
                await live.send_realtime_input(activity_end=types.ActivityEnd())
                nudge = (
                    f'Turno encerrado. turnId="{event.turn_id}". '
                    'Chame emit_feedback agora com este turnId.'
                )
                candidates = self._playbook_candidates_nudge(session)
                if candidates:
                    nudge = f'{nudge}\n{candidates}'
                prosody_nudge = self._prosody_context_nudge(session)
                if prosody_nudge:
                    nudge = f'{nudge}\n{prosody_nudge}'
                await live.send_realtime_input(text=nudge)
            return

    @staticmethod
    def _append_turn_pcm(session: _MeetingSession, pcm: bytes) -> None:
        if not pcm:
            return
        remaining = TURN_PCM_MAX_BYTES - len(session.turn_pcm)
        if remaining <= 0:
            return
        session.turn_pcm.extend(pcm[:remaining])

    def _schedule_prosody(
        self,
        session: _MeetingSession,
        turn_id: str,
        turn_pcm: bytes,
    ) -> None:
        if not turn_id or not turn_pcm:
            return

        async def _run() -> None:
            started = time.perf_counter()
            try:
                snapshot = await asyncio.to_thread(
                    analyze_turn_prosody,
                    turn_pcm,
                    sample_rate=session.sample_rate,
                    channels=session.channels,
                )
                session.prosody_by_turn[turn_id] = snapshot
                if snapshot.is_distinctive():
                    session.last_distinctive_prosody = snapshot
                # Bound cache so long meetings do not retain every turn.
                while len(session.prosody_by_turn) > _PROSODY_CACHE_MAX:
                    oldest = next(iter(session.prosody_by_turn))
                    session.prosody_by_turn.pop(oldest, None)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.info(
                    'prosody.analyze_ok | meeting=%s | turnId=%s | '
                    'durationMs=%s | meanRmsDbfs=%s | speechRatio=%.3f | '
                    'pauseCount=%s | longestPauseMs=%s | energy=%s | '
                    'hesitation=%s | distinctive=%s | analyzeMs=%.1f',
                    session.meeting_id,
                    turn_id,
                    snapshot.duration_ms,
                    snapshot.mean_rms_dbfs,
                    snapshot.speech_ratio,
                    snapshot.pause_count,
                    snapshot.longest_pause_ms,
                    snapshot.energy_level,
                    snapshot.hesitation_hint,
                    snapshot.is_distinctive(),
                    elapsed_ms,
                )
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.exception(
                    'prosody.analyze_failed | meeting=%s | turnId=%s | '
                    'pcmBytes=%s | analyzeMs=%.1f',
                    session.meeting_id,
                    turn_id,
                    len(turn_pcm),
                    elapsed_ms,
                )
            finally:
                session._prosody_tasks.pop(turn_id, None)

        # Cancel stale task for same turnId if any.
        prev = session._prosody_tasks.pop(turn_id, None)
        if prev is not None and not prev.done():
            prev.cancel()
        session._prosody_tasks[turn_id] = asyncio.create_task(
            _run(),
            name=f'prosody-{session.meeting_id}-{turn_id[:8]}',
        )

    @staticmethod
    def _prosody_context_nudge(session: _MeetingSession) -> str:
        snap = session.last_distinctive_prosody
        if snap is None:
            return ''
        return snap.nudge_line()

    async def _await_prosody(
        self,
        session: _MeetingSession,
        turn_id: str,
    ) -> Optional[ProsodySnapshot]:
        existing = session.prosody_by_turn.get(turn_id)
        if existing is not None:
            return existing
        task = session._prosody_tasks.get(turn_id)
        if task is None:
            return None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_PROSODY_PUBLISH_WAIT_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.info(
                'prosody.wait_timeout | meeting=%s | turnId=%s | '
                'waitMs=%s',
                session.meeting_id,
                turn_id,
                int(_PROSODY_PUBLISH_WAIT_S * 1000),
            )
            return session.prosody_by_turn.get(turn_id)
        except Exception:
            logger.exception(
                'prosody.wait_failed | meeting=%s | turnId=%s',
                session.meeting_id,
                turn_id,
            )
            return session.prosody_by_turn.get(turn_id)
        return session.prosody_by_turn.get(turn_id)

    async def _receive_loop(self, session: _MeetingSession, live: Any, types: Any) -> None:
        # google-genai AsyncSession.receive() ends after each turn_complete.
        # Outer while restarts so subsequent client turns still get tool calls.
        # See https://github.com/googleapis/python-genai/issues/1224
        while session.available:
            try:
                async for response in live.receive():
                    await self._handle_server_message(session, live, types, response)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    'Live receive loop error | meeting=%s | error=%s',
                    session.meeting_id,
                    exc,
                )
                break

    async def _handle_server_message(
        self,
        session: _MeetingSession,
        live: Any,
        types: Any,
        response: Any,
    ) -> None:
        usage = getattr(response, 'usage_metadata', None)
        if usage is not None:
            try:
                payload = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage)
            except Exception:
                payload = {}
            delta = session.cost.add_usage(payload)
            if delta:
                LIVE_COST_USD_TOTAL.inc(delta)
                LIVE_COST_USD_PER_MEETING.set(session.cost.total_usd)
            if session.cost.should_alert():
                logger.warning(
                    'Live cost alert | meeting=%s | usd=%.4f',
                    session.meeting_id,
                    session.cost.total_usd,
                )

        update = getattr(response, 'session_resumption_update', None)
        if update is not None:
            handle = getattr(update, 'new_handle', None) or getattr(
                update,
                'newHandle',
                None,
            )
            if handle:
                session.resumption_handle = str(handle)

        if getattr(response, 'data', None):
            nbytes = len(response.data)
            LIVE_UNEXPECTED_AUDIO_BYTES_TOTAL.inc(nbytes)
            session.cost.add_unexpected_audio(nbytes)

        server_content = getattr(response, 'server_content', None)
        if server_content is not None and getattr(server_content, 'turn_complete', False):
            session.awaiting_tool = False
            session.model_turn_done.set()

        tool_call = getattr(response, 'tool_call', None)
        if tool_call is None:
            return

        function_responses = []
        for fc in getattr(tool_call, 'function_calls', None) or []:
            name = getattr(fc, 'name', '') or ''
            args = getattr(fc, 'args', None) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if name == 'emit_feedback':
                await self._on_emit_feedback(session, args)

            function_responses.append(
                types.FunctionResponse(
                    id=getattr(fc, 'id', None) or str(uuid.uuid4()),
                    name=name or 'emit_feedback',
                    response={'result': 'ok'},
                ),
            )

        if function_responses:
            async with session.send_lock:
                await live.send_tool_response(function_responses=function_responses)
            # Release next VAD turn immediately — turn_complete often never
            # arrives for tool-only replies and used to cost a 2s start_timeout.
            session.awaiting_tool = False
            session.model_turn_done.set()

    async def _on_emit_feedback(
        self,
        session: _MeetingSession,
        args: dict[str, Any],
    ) -> None:
        pending = session.pending_turn
        turn_id = str(args.get('turnId') or args.get('turn_id') or '')
        if pending is not None:
            if not turn_id:
                turn_id = pending.turn_id
            speech_end_ms = pending.speech_end_ms
            meta = pending
            # Clear before publish so a second call for the same turn is ignored.
            if turn_id == pending.turn_id:
                session.pending_turn = None
        else:
            if not turn_id:
                return
            speech_end_ms = int(time.time() * 1000)
            meta = _PendingTurn(
                turn_id=turn_id,
                speech_end_ms=speech_end_ms,
                meeting_id=session.meeting_id,
                tenant_id=session.tenant_id,
                participant_id='meet-remote',
                participant_role='client',
            )

        latency_ms = max(0, int(time.time() * 1000) - speech_end_ms)
        LIVE_VAD_END_TO_TOOL_CALL_MS.observe(latency_ms)
        logger.info(
            '⏱️ LATENCY │ live.tool_call │ meeting=%s │ turnId=%s │ vadEndToToolMs=%s',
            session.meeting_id,
            turn_id,
            latency_ms,
        )

        # Seed next-turn RAG query (memory only).
        session.retrieve_query_hint = hint_from_emit_feedback_args(args)

        # Best-effort prosody merge; fail-open if analysis still running.
        prosody = await self._await_prosody(session, turn_id)
        if prosody is not None:
            logger.info(
                'prosody.merge_ok | meeting=%s | turnId=%s | '
                'energy=%s | hesitation=%s | meanRmsDbfs=%s',
                session.meeting_id,
                turn_id,
                prosody.energy_level,
                prosody.hesitation_hint,
                prosody.mean_rms_dbfs,
            )
        else:
            logger.info(
                'prosody.merge_miss | meeting=%s | turnId=%s | '
                'reason=not_ready_or_failed',
                session.meeting_id,
                turn_id,
            )

        # Publish on a worker thread — PublishDispatcher is sync/thread-safe.
        await asyncio.to_thread(
            self._publisher.publish_tool_call,
            meeting_id=meta.meeting_id,
            tenant_id=meta.tenant_id,
            participant_id=meta.participant_id,
            participant_role=meta.participant_role,
            args=args,
            speech_end_ms=speech_end_ms,
            turn_id=turn_id,
            prosody=prosody,
        )
        # Prefer next nudge from this turn once analyzed (may arrive late).
        if prosody is not None and prosody.is_distinctive():
            session.last_distinctive_prosody = prosody
        session.prosody_by_turn.pop(turn_id, None)

    def _playbook_candidates_nudge(self, session: _MeetingSession) -> str:
        index = session.playbook_index
        if index is None:
            return ''
        query = build_retrieve_query(
            context_summary=session.context_summary,
            retrieve_query_hint=session.retrieve_query_hint,
        )
        if not query.strip():
            return ''
        started = time.perf_counter()
        hits = index.retrieve(query, k=3)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 10.0 or hits:
            logger.info(
                'playbook.retrieve | meeting=%s | k=%s | keys=%s | ms=%.1f',
                session.meeting_id,
                len(hits),
                ','.join(h.key for h in hits),
                elapsed_ms,
            )
        return format_retrieve_nudge(hits)

    def _should_rotate(self, session: _MeetingSession) -> bool:
        if session.rotation_minutes <= 0:
            return False
        age_ms = int(time.time() * 1000) - session.opened_wall_ms
        return age_ms >= int(session.rotation_minutes * 60_000)
