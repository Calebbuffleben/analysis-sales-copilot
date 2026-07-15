"""Energy-based manual VAD for Gemini Live activity boundaries.

VAD only delimits turns (activity_start / activity_end). It never invents
feedback — that remains exclusively a Gemini tool call.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
from typing import List, Optional

from .audio_diagnostics import _SPEECH_SAMPLE_THRESHOLD


@dataclass(frozen=True)
class VadEvent:
    """One VAD state transition with PCM that should be forwarded."""

    kind: str  # 'activity_start' | 'audio' | 'activity_end'
    pcm: bytes = b''
    speech_end_ms: Optional[int] = None
    turn_id: str = ''


class ManualVad:
    """Frame energy VAD with a short prefix buffer.

    Ceiling: crude abs-sample heuristic (same threshold as audio_diagnostics).
    Upgrade path: WebRTC VAD or model-based endpointing if false ends rise.

    min_speech_ms: hold activity_start until enough speech accumulated so
    noise blips never open a Live turn (avoids empty micro tool calls).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_duration_ms: int = 250,
        prefix_ms: int = 120,
        speech_ratio_min: float = 0.08,
        min_speech_ms: int = 400,
    ) -> None:
        self._sample_rate = max(int(sample_rate), 1)
        self._channels = max(int(channels), 1)
        self._silence_duration_ms = max(int(silence_duration_ms), 50)
        self._prefix_bytes = int(
            self._sample_rate * self._channels * 2 * max(prefix_ms, 0) / 1000,
        )
        self._speech_ratio_min = max(0.0, min(float(speech_ratio_min), 1.0))
        self._min_speech_ms = max(0, int(min_speech_ms))
        self._speaking = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._prefix = bytearray()
        self._pending = bytearray()
        self._pending_turn_id = ''
        self._turn_seq = 0
        self._active_turn_id = ''

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def active_turn_id(self) -> str:
        return self._active_turn_id

    def reset(self) -> None:
        self._speaking = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._prefix.clear()
        self._pending.clear()
        self._pending_turn_id = ''
        self._active_turn_id = ''

    def push(self, pcm: bytes, timestamp_ms: int) -> List[VadEvent]:
        if not pcm:
            return []

        duration_ms = self._pcm_duration_ms(pcm)
        speechy = self._is_speech(pcm)
        events: List[VadEvent] = []

        if not self._speaking:
            if self._pending_turn_id:
                return self._push_pending(pcm, timestamp_ms, duration_ms, speechy)

            self._append_prefix(pcm)
            if not speechy:
                return events

            self._turn_seq += 1
            turn_id = f't{self._turn_seq}-{timestamp_ms}'
            prefix = bytes(self._prefix)
            self._prefix.clear()
            pending = bytearray()
            if prefix:
                pending.extend(prefix)
            if not prefix.endswith(pcm):
                pending.extend(pcm)
            speech_ms = self._pcm_duration_ms(bytes(pending)) if pending else duration_ms

            if speech_ms < self._min_speech_ms:
                self._pending = pending
                self._pending_turn_id = turn_id
                self._speech_ms = speech_ms
                self._silence_ms = 0
                return events

            return self._open_turn(turn_id, bytes(pending), already_includes_pcm=True)

        # Speaking
        events.append(
            VadEvent(kind='audio', pcm=pcm, turn_id=self._active_turn_id),
        )
        if speechy:
            self._speech_ms += duration_ms
            self._silence_ms = 0
            return events

        self._silence_ms += duration_ms
        if self._silence_ms >= self._silence_duration_ms:
            turn_id = self._active_turn_id
            events.append(
                VadEvent(
                    kind='activity_end',
                    speech_end_ms=timestamp_ms,
                    turn_id=turn_id,
                ),
            )
            self._speaking = False
            self._silence_ms = 0
            self._speech_ms = 0
            self._active_turn_id = ''
            self._prefix.clear()
        return events

    def force_end(self, timestamp_ms: int) -> List[VadEvent]:
        if self._pending_turn_id:
            # Promote short pending speech rather than drop mid-stream flush.
            events = self._open_turn(
                self._pending_turn_id,
                bytes(self._pending),
                already_includes_pcm=True,
            )
            self._pending.clear()
            self._pending_turn_id = ''
            if self._speaking:
                events.extend(self.force_end(timestamp_ms))
            return events
        if not self._speaking:
            return []
        turn_id = self._active_turn_id
        self._speaking = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._active_turn_id = ''
        self._prefix.clear()
        return [
            VadEvent(
                kind='activity_end',
                speech_end_ms=timestamp_ms,
                turn_id=turn_id,
            ),
        ]

    def _push_pending(
        self,
        pcm: bytes,
        timestamp_ms: int,
        duration_ms: int,
        speechy: bool,
    ) -> List[VadEvent]:
        if speechy:
            self._pending.extend(pcm)
            self._speech_ms += duration_ms
            self._silence_ms = 0
            if self._speech_ms >= self._min_speech_ms:
                turn_id = self._pending_turn_id
                buffered = bytes(self._pending)
                self._pending.clear()
                self._pending_turn_id = ''
                return self._open_turn(turn_id, buffered, already_includes_pcm=True)
            return []

        self._silence_ms += duration_ms
        if self._silence_ms >= self._silence_duration_ms:
            # Noise blip — discard without opening a Live turn.
            self._pending.clear()
            self._pending_turn_id = ''
            self._speech_ms = 0
            self._silence_ms = 0
            self._prefix.clear()
        return []

    def _open_turn(
        self,
        turn_id: str,
        pcm: bytes,
        *,
        already_includes_pcm: bool,
    ) -> List[VadEvent]:
        del already_includes_pcm  # pcm is the full buffer to forward
        self._speaking = True
        self._silence_ms = 0
        self._speech_ms = self._pcm_duration_ms(pcm) if pcm else 0
        self._active_turn_id = turn_id
        events: List[VadEvent] = [
            VadEvent(kind='activity_start', turn_id=turn_id),
        ]
        if pcm:
            events.append(VadEvent(kind='audio', pcm=pcm, turn_id=turn_id))
        return events

    def _append_prefix(self, pcm: bytes) -> None:
        if self._prefix_bytes <= 0:
            return
        self._prefix.extend(pcm)
        overflow = len(self._prefix) - self._prefix_bytes
        if overflow > 0:
            del self._prefix[:overflow]

    def _pcm_duration_ms(self, pcm: bytes) -> int:
        bytes_per_ms = max(self._sample_rate * self._channels * 2 / 1000.0, 1.0)
        return max(1, int(len(pcm) / bytes_per_ms))

    def _is_speech(self, pcm: bytes) -> bool:
        ch = self._channels
        if len(pcm) < 2 * ch:
            return False
        sample_count = len(pcm) // (2 * ch)
        if sample_count <= 0:
            return False
        arr = array.array('h')
        arr.frombytes(pcm[: sample_count * 2 * ch])
        mono = arr[::ch] if ch > 1 else arr
        if not mono:
            return False
        speech = sum(1 for s in mono if abs(s) >= _SPEECH_SAMPLE_THRESHOLD)
        return (speech / len(mono)) >= self._speech_ratio_min
