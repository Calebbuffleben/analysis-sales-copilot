"""Coordinate stable partial transcripts before early LLM analysis."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..text_analysis.types import TranscriptionChunk
from .utterance_completeness import (
    CompletenessVerdict,
    evaluate_utterance_completeness,
    text_fingerprint,
)

logger = logging.getLogger(__name__)

PartialReadyCallback = Callable[
    [str, TranscriptionChunk, dict[str, object]],
    None,
]


class FinalAction(str, Enum):
    SKIP = 'skip'
    PROCESS = 'process'
    RECONCILE = 'reconcile'


@dataclass
class PartialTurnConfig:
    enabled: bool = True
    stable_ms: int = 600
    word_stable_ms: int = 300
    growth_window_ms: int = 400
    min_words: int = 5
    cooldown_ms: int = 3000


@dataclass
class _StreamPartialState:
    last_text: str = ''
    last_text_wall_ms: int = 0
    last_growth_wall_ms: int = 0
    stable_text: str = ''
    stable_since_ms: int = 0
    last_word: str = ''
    last_word_stable_since_ms: int = 0
    last_dispatch_wall_ms: int = 0
    partial_published_fingerprint: str = ''
    partial_published_confidence: float = 0.0
    turn_open_wall_ms: int = 0


def _is_feedback_stream(meta: dict[str, object]) -> bool:
    role = str(meta.get('participant_role') or '').strip().lower()
    track = str(meta.get('track') or '').strip().lower()
    if role == 'host':
        return False
    if role in {'participant', 'client'}:
        return True
    return track == 'tab-audio'


class PartialTurnCoordinator:
    """Track partial stability and invoke the pipeline when utterance looks complete."""

    def __init__(
        self,
        config: PartialTurnConfig,
        on_partial_ready: PartialReadyCallback,
    ) -> None:
        self._config = config
        self._on_partial_ready = on_partial_ready
        self._lock = threading.RLock()
        self._streams: dict[str, _StreamPartialState] = {}

    def on_turn_audio_start(self, stream_key: str, wall_ms: int) -> None:
        with self._lock:
            state = self._streams.setdefault(stream_key, _StreamPartialState())
            state.turn_open_wall_ms = wall_ms

    def handle_partial(
        self,
        stream_key: str,
        transcript: str,
        wall_ms: int,
        meta: dict[str, object],
    ) -> None:
        if not self._config.enabled or not _is_feedback_stream(meta):
            return

        text = (transcript or '').strip()
        if not text:
            return

        with self._lock:
            state = self._streams.setdefault(stream_key, _StreamPartialState())
            if text != state.last_text:
                if len(text) > len(state.last_text):
                    state.last_growth_wall_ms = wall_ms
                state.last_text = text
                state.last_text_wall_ms = wall_ms

            word = text.rstrip('.,!?;:…').split()[-1].lower() if text.split() else ''
            if word != state.last_word:
                state.last_word = word
                state.last_word_stable_since_ms = wall_ms

            if text != state.stable_text:
                state.stable_text = text
                state.stable_since_ms = wall_ms

            stable_ms = wall_ms - state.stable_since_ms
            word_stable_ms = wall_ms - state.last_word_stable_since_ms
            if stable_ms < self._config.stable_ms:
                return
            if word_stable_ms < self._config.word_stable_ms:
                return

            if (
                self._config.cooldown_ms > 0
                and state.last_dispatch_wall_ms > 0
                and (wall_ms - state.last_dispatch_wall_ms) < self._config.cooldown_ms
            ):
                return

            fingerprint = text_fingerprint(text)
            if fingerprint and fingerprint == state.partial_published_fingerprint:
                return

            completeness = evaluate_utterance_completeness(
                text,
                min_words=self._config.min_words,
                now_wall_ms=wall_ms,
                last_growth_wall_ms=state.last_growth_wall_ms,
                growth_window_ms=self._config.growth_window_ms,
            )
            if completeness.verdict != CompletenessVerdict.READY:
                logger.debug(
                    'partial blocked by completeness | stream=%s | reason=%s | text=%r',
                    stream_key,
                    completeness.reason,
                    text[:80],
                )
                return

            state.last_dispatch_wall_ms = wall_ms
            turn_open_ms = (
                max(0, wall_ms - state.turn_open_wall_ms)
                if state.turn_open_wall_ms
                else None
            )

        turn_start_ms = int(meta.get('turn_start_ms') or wall_ms)
        chunk = TranscriptionChunk(
            meeting_id=str(meta.get('meeting_id') or ''),
            participant_id=str(meta.get('participant_id') or ''),
            track=str(meta.get('track') or ''),
            text=text,
            confidence=float(meta.get('transcript_confidence') or 0.0),
            timestamp_ms=wall_ms,
            window_start_ms=turn_start_ms,
            window_end_ms=wall_ms,
            tenant_id=str(meta.get('tenant_id') or ''),
            participant_role=str(meta.get('participant_role') or ''),
        )
        extra = {
            'transcriptSource': 'partial',
            'completenessReason': completeness.reason,
            'turnOpenMs': turn_open_ms,
        }
        logger.info(
            'partial stable ready | stream=%s | reason=%s | chars=%s',
            stream_key,
            completeness.reason,
            len(text),
        )
        self._on_partial_ready(stream_key, chunk, extra)

    def plan_final_action(self, stream_key: str, final_text: str) -> FinalAction:
        fingerprint = text_fingerprint(final_text)
        with self._lock:
            state = self._streams.get(stream_key)
            if state is None or not state.partial_published_fingerprint:
                return FinalAction.PROCESS
            if fingerprint == state.partial_published_fingerprint:
                return FinalAction.SKIP
            return FinalAction.RECONCILE

    def note_partial_published(
        self,
        stream_key: str,
        text: str,
        confidence: float,
    ) -> None:
        with self._lock:
            state = self._streams.setdefault(stream_key, _StreamPartialState())
            state.partial_published_fingerprint = text_fingerprint(text)
            state.partial_published_confidence = confidence

    def partial_published_confidence(self, stream_key: str) -> float:
        with self._lock:
            state = self._streams.get(stream_key)
            return state.partial_published_confidence if state else 0.0

    def reset_turn(self, stream_key: str) -> None:
        with self._lock:
            self._streams.pop(stream_key, None)
