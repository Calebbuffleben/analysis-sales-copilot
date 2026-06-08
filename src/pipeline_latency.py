"""Per-turn pipeline latency milestones (AssemblyAI → Gemini).

Emits prominent ⏱️ LATENCY logs at four checkpoints with delta-ms between stages.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

LATENCY_MARKER = '⏱️ LATENCY'

STAGE_ORDER: dict[str, int] = {
    'assemblyai.audio_sent': 1,
    'assemblyai.transcript_received': 2,
    'gemini.prompt_sent': 3,
    'gemini.response_received': 4,
}

STAGE_LABEL: dict[str, str] = {
    'assemblyai.audio_sent': '① assemblyai.audio_sent',
    'assemblyai.transcript_received': '② assemblyai.transcript_received',
    'gemini.prompt_sent': '③ gemini.prompt_sent',
    'gemini.response_received': '④ gemini.response_received',
}


@dataclass(frozen=True)
class LatencyTraceContext:
    """Correlation handle shared across AssemblyAI → Gemini."""

    trace_id: str
    meeting_id: str
    participant_id: str
    window_end_ms: int


@dataclass
class _StreamTurnState:
    turn_start_wall_ms: int = 0
    last_audio_sent_wall_ms: int = 0
    turn_bytes: int = 0
    chunk_sends: int = 0


@dataclass
class _TraceTimeline:
    origin_wall_ms: int
    last_wall_ms: int
    last_stage: str = ''
    stages: dict[str, int] = field(default_factory=dict)


_lock = threading.RLock()
_stream_turns: dict[str, _StreamTurnState] = {}
_trace_timelines: dict[str, _TraceTimeline] = {}


def _wall_ms() -> int:
    return int(time.time() * 1000)


def _format_human_line(
    stage: str,
    *,
    trace_id: str,
    meeting_id: str,
    participant_id: str,
    delta_prev_ms: Optional[int],
    delta_origin_ms: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    order = STAGE_ORDER.get(stage, 0)
    label = STAGE_LABEL.get(stage, stage)
    prev = (
        f'+{delta_prev_ms}ms desde anterior'
        if delta_prev_ms is not None
        else 'início do turno'
    )
    parts = [
        LATENCY_MARKER,
        f'│ {order}/4 {label}',
        f'│ {prev}',
        f'│ +{delta_origin_ms}ms total',
        f'│ traceId={trace_id}',
        f'│ meeting={meeting_id}',
        f'│ participant={participant_id}',
    ]
    if extra:
        for key, value in extra.items():
            if value is not None:
                parts.append(f'│ {key}={value}')
    return ' '.join(parts)


def _mark_stage(trace_id: str, stage: str, wall_ms: int) -> tuple[Optional[int], int]:
    with _lock:
        timeline = _trace_timelines.get(trace_id)
        if timeline is None:
            timeline = _TraceTimeline(origin_wall_ms=wall_ms, last_wall_ms=wall_ms)
            _trace_timelines[trace_id] = timeline
        delta_prev = (
            max(0, wall_ms - timeline.last_wall_ms)
            if timeline.last_stage
            else None
        )
        delta_origin = max(0, wall_ms - timeline.origin_wall_ms)
        timeline.last_wall_ms = wall_ms
        timeline.last_stage = stage
        timeline.stages[stage] = wall_ms
        return delta_prev, delta_origin


def _log_milestone(
    logger: logging.Logger,
    stage: str,
    *,
    trace_id: str,
    meeting_id: str,
    participant_id: str,
    window_end_ms: int,
    wall_ms: Optional[int] = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Optional[int], int]:
    now_ms = wall_ms if wall_ms is not None else _wall_ms()
    delta_prev, delta_origin = _mark_stage(trace_id, stage, now_ms)
    payload: dict[str, Any] = {
        'stage': f'python.latency.{stage}',
        'traceId': trace_id,
        'meetingId': meeting_id,
        'participantId': participant_id,
        'windowEndMs': int(window_end_ms),
        'milestone': stage,
        'milestoneOrder': STAGE_ORDER.get(stage),
        'deltaPrevMs': delta_prev,
        'deltaOriginMs': delta_origin,
        'wallMs': now_ms,
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
    logger.info(
        _format_human_line(
            stage,
            trace_id=trace_id,
            meeting_id=meeting_id,
            participant_id=participant_id,
            delta_prev_ms=delta_prev,
            delta_origin_ms=delta_origin,
            extra=extra,
        ),
    )
    logger.info(json.dumps(payload, default=str, separators=(',', ':')))
    return delta_prev, delta_origin


def note_assemblyai_audio_sent(
    logger: logging.Logger,
    stream_key: str,
    *,
    chunk_bytes: int,
    turn_bytes: int,
    is_turn_start: bool,
) -> None:
    """Track PCM chunks sent to AssemblyAI; log milestone ① on each send."""
    now_ms = _wall_ms()
    with _lock:
        state = _stream_turns.get(stream_key)
        if state is None or is_turn_start:
            state = _StreamTurnState(
                turn_start_wall_ms=now_ms,
                last_audio_sent_wall_ms=now_ms,
                turn_bytes=chunk_bytes,
                chunk_sends=1,
            )
            _stream_turns[stream_key] = state
        else:
            state.last_audio_sent_wall_ms = now_ms
            state.turn_bytes = turn_bytes
            state.chunk_sends += 1

    logger.info(
        '%s │ 1/4 ① assemblyai.audio_sent │ stream=%s │ chunkBytes=%s │ '
        'turnBytes=%s │ chunk#=%s │ turnStart=%s',
        LATENCY_MARKER,
        stream_key,
        chunk_bytes,
        turn_bytes,
        state.chunk_sends,
        is_turn_start,
    )


def pop_stream_turn_state(stream_key: str) -> Optional[_StreamTurnState]:
    with _lock:
        return _stream_turns.pop(stream_key, None)


def log_assemblyai_transcript_received(
    logger: logging.Logger,
    ctx: LatencyTraceContext,
    *,
    stream_key: str,
    since_last_audio_ms: int,
    turn_bytes: int,
    audio_chunks_sent: int,
    transcript_chars: int,
    turn_audio_ms: Optional[int] = None,
    last_audio_sent_wall_ms: Optional[int] = None,
) -> None:
    """Milestone ② — finalized turn from AssemblyAI."""
    if last_audio_sent_wall_ms is not None:
        _mark_stage(ctx.trace_id, 'assemblyai.audio_sent', last_audio_sent_wall_ms)
    _log_milestone(
        logger,
        'assemblyai.transcript_received',
        trace_id=ctx.trace_id,
        meeting_id=ctx.meeting_id,
        participant_id=ctx.participant_id,
        window_end_ms=ctx.window_end_ms,
        extra={
            'streamKey': stream_key,
            'sttLatencyMs': since_last_audio_ms,
            'turnBytes': turn_bytes,
            'audioChunksSent': audio_chunks_sent,
            'transcriptChars': transcript_chars,
            'turnAudioMs': turn_audio_ms,
        },
    )


def log_gemini_prompt_sent(
    logger: logging.Logger,
    ctx: LatencyTraceContext,
    *,
    prompt_chars: int,
    speaker_role: str,
    provider: str = 'gemini',
) -> int:
    """Milestone ③ — prompt dispatched to LLM. Returns wall_ms for round-trip."""
    now_ms = _wall_ms()
    _log_milestone(
        logger,
        'gemini.prompt_sent',
        trace_id=ctx.trace_id,
        meeting_id=ctx.meeting_id,
        participant_id=ctx.participant_id,
        window_end_ms=ctx.window_end_ms,
        wall_ms=now_ms,
        extra={
            'promptChars': prompt_chars,
            'speakerRole': speaker_role,
            'llmProvider': provider,
        },
    )
    return now_ms


def log_gemini_response_received(
    logger: logging.Logger,
    ctx: LatencyTraceContext,
    *,
    prompt_sent_wall_ms: int,
    response_chars: int,
    llm_round_trip_ms: int,
    has_feedback: bool,
    confidence: Optional[float] = None,
    provider: str = 'gemini',
) -> None:
    """Milestone ④ — LLM response parsed; prints end-to-end summary."""
    _log_milestone(
        logger,
        'gemini.response_received',
        trace_id=ctx.trace_id,
        meeting_id=ctx.meeting_id,
        participant_id=ctx.participant_id,
        window_end_ms=ctx.window_end_ms,
        extra={
            'responseChars': response_chars,
            'llmRoundTripMs': llm_round_trip_ms,
            'hasFeedback': has_feedback,
            'confidence': round(confidence, 3) if confidence is not None else None,
            'llmProvider': provider,
        },
    )

    with _lock:
        timeline = _trace_timelines.get(ctx.trace_id)

    if timeline is None:
        return

    def _seg(start: str, end: str) -> Optional[int]:
        a = timeline.stages.get(start)
        b = timeline.stages.get(end)
        if a is None or b is None:
            return None
        return max(0, b - a)

    stt_ms = _seg('assemblyai.audio_sent', 'assemblyai.transcript_received')
    pre_llm_ms = _seg('assemblyai.transcript_received', 'gemini.prompt_sent')
    llm_ms = _seg('gemini.prompt_sent', 'gemini.response_received')
    total_ms = max(0, _wall_ms() - timeline.origin_wall_ms)

    logger.info(
        '%s │ RESUMO │ traceId=%s │ meeting=%s │ '
        'STT(①→②)=%sms │ fila+prep(②→③)=%sms │ LLM(③→④)=%sms │ total≈%sms',
        LATENCY_MARKER,
        ctx.trace_id,
        ctx.meeting_id,
        stt_ms if stt_ms is not None else '?',
        pre_llm_ms if pre_llm_ms is not None else '?',
        llm_ms if llm_ms is not None else llm_round_trip_ms,
        total_ms,
    )

    with _lock:
        _trace_timelines.pop(ctx.trace_id, None)


def clear_trace(trace_id: str) -> None:
    with _lock:
        _trace_timelines.pop(trace_id, None)
