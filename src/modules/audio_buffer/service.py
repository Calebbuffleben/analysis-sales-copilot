"""High-level service interface for audio buffering with sliding windows.

This module will coordinate:
- Per-stream sliding-window buffers for raw PCM audio, backed by CircularBuffer.
- Integration points where AudioService can push incoming WAV chunks
  (after header stripping) into the buffers.
- A SlidingWindowWorker that decides *when* olhar para a janela atual e
  notificar o pipeline de transcrição (TranscriptionPipelineService).

Implementation TODOs (arquitetura alvo):
- Definir uma classe AudioBufferService com responsabilidades claras:
  - Gerenciar um CircularBuffer por stream_key (meeting:participant:track).
  - Expor APIs de alto nível:
    - push(stream_key, wav_data, sample_rate, channels, timestamp_ms, sequence)
    - get_window(stream_key) -> Optional[bytes]
    - end_stream(stream_key)
  - Fazer strip do header WAV para obter PCM antes de alimentar o buffer.
- Integrar com SlidingWindowWorker:
  - Injetar um SlidingWindowWorker.
  - Após cada push, chamar worker.on_chunk_appended(stream_key, timestamp_ms).
- Interagir com TranscriptionPipelineService de forma indireta:
  - TranscriptionPipelineService registra um callback em SlidingWindowWorker.
  - Quando o worker decidir que a janela está pronta, ele chamará esse callback,
    passando (stream_key, window_pcm_bytes, meta).
- Ligar AudioBufferService ao AudioService:
  - AudioService chamará AudioBufferService.push(...) em process_chunk().
  - AudioService chamará AudioBufferService.end_stream(...) em end_stream().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .circular_buffer import CircularBuffer
from .sliding_worker import SlidingWindowWorker, WindowReadyCallback

WAV_HEADER_BYTES = 44


@dataclass
class AudioBufferState:
    """Per-stream state for sliding-window buffering."""

    buffer: CircularBuffer
    meeting_id: str
    participant_id: str
    track: str
    sample_rate: int
    channels: int
    last_timestamp_ms: int
    last_sequence: int
    tenant_id: str = ''


class AudioBufferService:
    """Manage PCM sliding windows per stream and notify the worker."""

    def __init__(
        self,
        worker: Optional[SlidingWindowWorker] = None,
        window_seconds: float = 10.0,
    ) -> None:
        self._window_seconds = window_seconds
        self._worker = worker or SlidingWindowWorker()
        self._states: Dict[str, AudioBufferState] = {}
        self._worker.set_window_reader(self.get_window_data)

    def register_window_callback(self, callback: WindowReadyCallback) -> None:
        """Register the callback that consumes ready windows."""
        self._worker.register_window_callback(callback)

    def push(
        self,
        stream_key: str,
        wav_data: bytes,
        sample_rate: int,
        channels: int,
        timestamp_ms: int,
        sequence: int,
        tenant_id: str = '',
    ) -> None:
        """Append audio to the stream buffer and notify the worker."""
        pcm_data = self._extract_pcm(wav_data)
        if not pcm_data:
            return

        state = self._get_or_create_state(
            stream_key=stream_key,
            sample_rate=sample_rate,
            channels=channels,
            timestamp_ms=timestamp_ms,
            sequence=sequence,
            tenant_id=tenant_id,
        )
        state.buffer.append(pcm_data)
        state.last_timestamp_ms = timestamp_ms
        state.last_sequence = sequence
        if tenant_id and not state.tenant_id:
            state.tenant_id = tenant_id
        self._worker.on_chunk_appended(stream_key, timestamp_ms)

    def get_window(self, stream_key: str) -> Optional[bytes]:
        """Return the latest PCM window for a stream."""
        state = self._states.get(stream_key)
        if not state:
            return None
        return state.buffer.read_all()

    def get_window_data(
        self,
        stream_key: str,
    ) -> Optional[Tuple[bytes, Dict[str, object]]]:
        """Return the latest PCM window plus metadata for a stream."""
        state = self._states.get(stream_key)
        if not state:
            return None

        window_pcm = state.buffer.read_all()
        if not window_pcm:
            return None

        bytes_per_second = state.sample_rate * state.channels * 2
        window_duration_ms = int((len(window_pcm) / max(bytes_per_second, 1)) * 1000)
        window_end_ms = state.last_timestamp_ms
        window_start_ms = max(0, window_end_ms - window_duration_ms)

        return window_pcm, {
            'stream_key': stream_key,
            'meeting_id': state.meeting_id,
            'participant_id': state.participant_id,
            'track': state.track,
            'sample_rate': state.sample_rate,
            'channels': state.channels,
            'sequence': state.last_sequence,
            'window_start_ms': window_start_ms,
            'window_end_ms': window_end_ms,
            'tenant_id': state.tenant_id,
        }

    def end_stream(self, stream_key: str) -> None:
        """Flush and cleanup a stream buffer."""
        if stream_key in self._states:
            self._worker.flush_stream(stream_key)
            self._worker.clear_stream_state(stream_key)
            self._states.pop(stream_key, None)

    def _get_or_create_state(
        self,
        stream_key: str,
        sample_rate: int,
        channels: int,
        timestamp_ms: int,
        sequence: int,
        tenant_id: str = '',
    ) -> AudioBufferState:
        state = self._states.get(stream_key)
        if state:
            return state

        meeting_id, participant_id, track = self._split_stream_key(stream_key)
        capacity_bytes = int(sample_rate * channels * 2 * self._window_seconds)
        state = AudioBufferState(
            buffer=CircularBuffer(capacity_bytes),
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
            last_timestamp_ms=timestamp_ms,
            last_sequence=sequence,
            tenant_id=tenant_id,
        )
        self._states[stream_key] = state
        return state

    def _extract_pcm(self, wav_data: bytes) -> bytes:
        """Strip a WAV header when present and return raw PCM."""
        if len(wav_data) >= WAV_HEADER_BYTES and wav_data[:4] == b'RIFF':
            return wav_data[WAV_HEADER_BYTES:]
        return wav_data

    def _split_stream_key(self, stream_key: str) -> Tuple[str, str, str]:
        """Split a stream key into meeting_id, participant_id and track."""
        parts = stream_key.split(':', 2)
        if len(parts) != 3:
            raise ValueError(f'Invalid stream key: {stream_key}')
        return parts[0], parts[1], parts[2]

