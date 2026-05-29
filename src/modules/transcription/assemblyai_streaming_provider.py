"""AssemblyAI Streaming Speech-to-Text provider.

The provider owns one WebSocket session per audio stream and emits finalized
turns into the existing text-analysis pipeline as ``TranscriptionChunk``s.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlencode

from ...metrics.realtime_metrics import (
    ASSEMBLYAI_AUDIO_BYTES_SENT_TOTAL,
    ASSEMBLYAI_EMPTY_TURNS_TOTAL,
    ASSEMBLYAI_ERRORS_TOTAL,
    ASSEMBLYAI_FINAL_TURNS_TOTAL,
    ASSEMBLYAI_RECONNECTS_TOTAL,
    ASSEMBLYAI_SESSIONS_OPEN,
    ASSEMBLYAI_SESSIONS_STARTED_TOTAL,
    ASSEMBLYAI_SESSIONS_TERMINATED_TOTAL,
    ASSEMBLYAI_TURN_LATENCY_MS,
    ASSEMBLYAI_TURNS_TOTAL,
)
from ..audio_buffer.audio_diagnostics import compute_pcm_window_stats
from ..audio_buffer.service import WAV_HEADER_BYTES
from ..text_analysis.types import TranscriptionChunk

logger = logging.getLogger(__name__)

FinalTranscriptCallback = Callable[[str, TranscriptionChunk, dict[str, object]], None]


@dataclass
class AssemblyAiStreamConfig:
    """Runtime configuration for AssemblyAI Universal Streaming."""

    api_key: str
    api_host: str = 'streaming.assemblyai.com'
    speech_model: str = 'u3-rt-pro'
    sample_rate: int = 16000
    format_turns: bool = True
    continuous_partials: bool = False
    stream_idle_timeout_ms: int = 30_000
    reconnect_limit: int = 2
    connect_timeout_seconds: float = 10.0
    termination_timeout_seconds: float = 2.0
    end_of_turn_confidence_threshold: Optional[float] = None
    min_turn_silence_ms: Optional[int] = None
    max_turn_silence_ms: Optional[int] = None
    vad_threshold: Optional[float] = None
    keyterms_prompt: Optional[str] = None


@dataclass
class _AssemblyAiSession:
    stream_key: str
    meta: dict[str, object]
    sample_rate: int
    channels: int
    ws_app: Any
    thread: threading.Thread
    open_event: threading.Event = field(default_factory=threading.Event)
    termination_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    current_turn_pcm: bytearray = field(default_factory=bytearray)
    current_turn_start_ms: Optional[int] = None
    last_audio_timestamp_ms: int = 0
    last_audio_wall_ms: int = 0
    bytes_sent: int = 0
    reconnects: int = 0
    closed: bool = False


class AssemblyAiStreamingProvider:
    """Manage AssemblyAI WebSocket sessions and map final turns to chunks."""

    def __init__(
        self,
        config: AssemblyAiStreamConfig,
        on_final_transcript: FinalTranscriptCallback,
    ) -> None:
        self._config = config
        self._on_final_transcript = on_final_transcript
        self._sessions: dict[str, _AssemblyAiSession] = {}
        self._lock = threading.RLock()
        self._callback_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix='assemblyai-final-turn',
        )
        self._idle_stop = threading.Event()
        self._idle_thread = threading.Thread(
            target=self._idle_cleanup_loop,
            name='assemblyai-idle-cleanup',
            daemon=True,
        )
        self._idle_thread.start()

    def start_stream(
        self,
        stream_key: str,
        meta: dict[str, object],
    ) -> None:
        """Ensure the AssemblyAI session for a stream is open."""
        with self._lock:
            existing = self._sessions.get(stream_key)
            if existing and not existing.closed:
                return
            session = self._create_session(stream_key, meta)
            self._sessions[stream_key] = session

        if not session.open_event.wait(self._config.connect_timeout_seconds):
            self.end_stream(stream_key)
            raise RuntimeError(
                f'AssemblyAI stream did not open within '
                f'{self._config.connect_timeout_seconds:.1f}s for {stream_key}',
            )

    def send_audio(
        self,
        stream_key: str,
        audio_data: bytes,
        meta: dict[str, object],
    ) -> None:
        """Send raw PCM bytes for a stream, stripping WAV headers if present."""
        pcm_data = self._extract_pcm(audio_data)
        if not pcm_data:
            return

        session = self._get_or_start_session(stream_key, meta)
        with session.lock:
            now_ms = int(time.time() * 1000)
            timestamp_ms = int(meta.get('timestamp_ms', now_ms) or now_ms)
            if session.current_turn_start_ms is None:
                duration_ms = self._duration_ms(
                    pcm_data,
                    sample_rate=session.sample_rate,
                    channels=session.channels,
                )
                session.current_turn_start_ms = max(0, timestamp_ms - duration_ms)
            session.current_turn_pcm.extend(pcm_data)
            session.last_audio_timestamp_ms = timestamp_ms
            session.last_audio_wall_ms = now_ms
            session.bytes_sent += len(pcm_data)

        try:
            session.ws_app.send(pcm_data, opcode=self._binary_opcode())
            ASSEMBLYAI_AUDIO_BYTES_SENT_TOTAL.inc(len(pcm_data))
        except Exception as exc:
            ASSEMBLYAI_ERRORS_TOTAL.inc()
            logger.warning(
                'AssemblyAI send failed | stream_key=%s | error=%s',
                stream_key,
                exc,
            )
            self._attempt_reconnect(stream_key, meta)
            session = self._get_or_start_session(stream_key, meta)
            session.ws_app.send(pcm_data, opcode=self._binary_opcode())
            ASSEMBLYAI_AUDIO_BYTES_SENT_TOTAL.inc(len(pcm_data))

    def end_stream(self, stream_key: str) -> None:
        """Terminate and remove a single AssemblyAI session."""
        with self._lock:
            session = self._sessions.pop(stream_key, None)
        if session is None:
            return
        self._terminate_session(session)

    def close_all(self) -> None:
        """Terminate all open sessions and stop background workers."""
        with self._lock:
            keys = list(self._sessions.keys())
        for key in keys:
            self.end_stream(key)
        self._idle_stop.set()
        self._idle_thread.join(timeout=1.0)
        self._callback_executor.shutdown(wait=True, cancel_futures=False)

    def _get_or_start_session(
        self,
        stream_key: str,
        meta: dict[str, object],
    ) -> _AssemblyAiSession:
        with self._lock:
            session = self._sessions.get(stream_key)
        if session is None or session.closed:
            self.start_stream(stream_key, meta)
            with self._lock:
                session = self._sessions[stream_key]
        if not session.open_event.wait(self._config.connect_timeout_seconds):
            raise RuntimeError(f'AssemblyAI stream is not open for {stream_key}')
        return session

    def _create_session(
        self,
        stream_key: str,
        meta: dict[str, object],
    ) -> _AssemblyAiSession:
        try:
            import websocket
        except ImportError as error:
            raise RuntimeError(
                'websocket-client is required for AssemblyAI streaming. '
                'Install python-service requirements before starting.',
            ) from error

        sample_rate = int(meta.get('sample_rate') or self._config.sample_rate)
        channels = max(int(meta.get('channels') or 1), 1)
        ws_app = websocket.WebSocketApp(
            self._build_url(sample_rate),
            header={'Authorization': self._config.api_key},
            on_open=lambda _ws: self._on_open(stream_key),
            on_message=lambda _ws, message: self._on_message(stream_key, message),
            on_error=lambda _ws, error: self._on_error(stream_key, error),
            on_close=lambda _ws, code, reason: self._on_close(
                stream_key,
                code,
                reason,
            ),
        )
        session = _AssemblyAiSession(
            stream_key=stream_key,
            meta=dict(meta),
            sample_rate=sample_rate,
            channels=channels,
            ws_app=ws_app,
            thread=threading.Thread(
                target=ws_app.run_forever,
                name=f'assemblyai-{stream_key}',
                daemon=True,
            ),
        )
        session.thread.start()
        ASSEMBLYAI_SESSIONS_STARTED_TOTAL.inc()
        ASSEMBLYAI_SESSIONS_OPEN.inc()
        logger.info(
            'AssemblyAI stream opening | stream_key=%s | sample_rate=%s | '
            'channels=%s | model=%s | format_turns=%s',
            stream_key,
            sample_rate,
            channels,
            self._config.speech_model,
            self._config.format_turns,
        )
        return session

    def _build_url(self, sample_rate: int) -> str:
        params: dict[str, object] = {
            'speech_model': self._config.speech_model,
            'sample_rate': sample_rate,
            'format_turns': self._config.format_turns,
        }
        if self._config.continuous_partials:
            params['continuous_partials'] = True
        if self._config.end_of_turn_confidence_threshold is not None:
            params['end_of_turn_confidence_threshold'] = (
                self._config.end_of_turn_confidence_threshold
            )
        if self._config.min_turn_silence_ms is not None:
            params['min_turn_silence'] = self._config.min_turn_silence_ms
        if self._config.max_turn_silence_ms is not None:
            params['max_turn_silence'] = self._config.max_turn_silence_ms
        if self._config.vad_threshold is not None:
            params['vad_threshold'] = self._config.vad_threshold
        if self._config.keyterms_prompt:
            params['keyterms_prompt'] = self._config.keyterms_prompt

        encoded = urlencode(
            {
                key: str(value).lower() if isinstance(value, bool) else value
                for key, value in params.items()
            },
        )
        return f'wss://{self._config.api_host}/v3/ws?{encoded}'

    def _on_open(self, stream_key: str) -> None:
        session = self._get_session(stream_key)
        if session is None:
            return
        session.open_event.set()
        logger.info('AssemblyAI stream opened | stream_key=%s', stream_key)

    def _on_message(self, stream_key: str, message: object) -> None:
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            payload = json.loads(str(message))
        except Exception as exc:
            ASSEMBLYAI_ERRORS_TOTAL.inc()
            logger.warning(
                'AssemblyAI message parse failed | stream_key=%s | error=%s',
                stream_key,
                exc,
            )
            return

        msg_type = payload.get('type')
        if msg_type == 'Begin':
            logger.info(
                'AssemblyAI session began | stream_key=%s | session_id=%s',
                stream_key,
                payload.get('id'),
            )
            return
        if msg_type == 'Termination':
            session = self._get_session(stream_key)
            if session:
                session.termination_event.set()
            logger.info(
                'AssemblyAI session terminated | stream_key=%s | audio_s=%s | session_s=%s',
                stream_key,
                payload.get('audio_duration_seconds'),
                payload.get('session_duration_seconds'),
            )
            return
        if msg_type != 'Turn':
            return

        ASSEMBLYAI_TURNS_TOTAL.inc()
        if not bool(payload.get('end_of_turn')):
            return
        self._handle_final_turn(stream_key, payload)

    def _handle_final_turn(self, stream_key: str, payload: dict[str, Any]) -> None:
        ASSEMBLYAI_FINAL_TURNS_TOTAL.inc()
        transcript = str(payload.get('transcript') or '').strip()
        session = self._get_session(stream_key)
        if session is None:
            return
        if not transcript:
            ASSEMBLYAI_EMPTY_TURNS_TOTAL.inc()
            self._reset_turn_audio(session)
            return

        with session.lock:
            turn_pcm = bytes(session.current_turn_pcm)
            window_start_ms = int(
                session.current_turn_start_ms
                or session.last_audio_timestamp_ms
                or int(time.time() * 1000),
            )
            window_end_ms = int(
                session.last_audio_timestamp_ms
                or window_start_ms
                or int(time.time() * 1000),
            )
            wall_end_ms = session.last_audio_wall_ms
            session.current_turn_pcm.clear()
            session.current_turn_start_ms = None

        if wall_end_ms:
            ASSEMBLYAI_TURN_LATENCY_MS.observe(
                max(0.0, float(int(time.time() * 1000) - wall_end_ms)),
            )

        words = payload.get('words')
        confidence = self._confidence_from_words(words)
        stats = compute_pcm_window_stats(
            turn_pcm,
            sample_rate=session.sample_rate,
            channels=session.channels,
        )
        chunk = TranscriptionChunk(
            meeting_id=str(session.meta.get('meeting_id') or ''),
            participant_id=str(session.meta.get('participant_id') or ''),
            track=str(session.meta.get('track') or ''),
            text=transcript,
            confidence=confidence,
            timestamp_ms=window_end_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            tenant_id=str(session.meta.get('tenant_id') or ''),
        )
        extra_stats: dict[str, object] = {
            'samples_count': int(stats.get('samples_count') or 0),
            'speech_count': int(stats.get('speech_count') or 0),
            'mean_rms_dbfs': stats.get('mean_rms_dbfs'),
            'assemblyai_turn_order': payload.get('turn_order'),
            'assemblyai_turn_is_formatted': payload.get('turn_is_formatted'),
            'assemblyai_end_of_turn_confidence': payload.get(
                'end_of_turn_confidence',
            ),
        }
        self._callback_executor.submit(
            self._on_final_transcript,
            stream_key,
            chunk,
            extra_stats,
        )

    def _on_error(self, stream_key: str, error: object) -> None:
        ASSEMBLYAI_ERRORS_TOTAL.inc()
        logger.warning(
            'AssemblyAI stream error | stream_key=%s | error=%s',
            stream_key,
            error,
        )

    def _on_close(
        self,
        stream_key: str,
        code: object,
        reason: object,
    ) -> None:
        session = self._get_session(stream_key)
        if session is None:
            return
        self._mark_closed(session)
        logger.info(
            'AssemblyAI stream closed | stream_key=%s | code=%s | reason=%s',
            stream_key,
            code,
            reason,
        )

    def _attempt_reconnect(
        self,
        stream_key: str,
        meta: dict[str, object],
    ) -> None:
        session = self._get_session(stream_key)
        if session is None:
            return
        if session.reconnects >= self._config.reconnect_limit:
            raise RuntimeError(
                f'AssemblyAI reconnect limit exceeded for {stream_key}',
            )
        session.reconnects += 1
        ASSEMBLYAI_RECONNECTS_TOTAL.inc()
        self.end_stream(stream_key)
        self.start_stream(stream_key, meta)

    def _terminate_session(self, session: _AssemblyAiSession) -> None:
        try:
            if session.open_event.is_set() and not session.closed:
                session.ws_app.send(json.dumps({'type': 'Terminate'}))
                session.termination_event.wait(
                    timeout=self._config.termination_timeout_seconds,
                )
        except Exception as exc:
            logger.debug(
                'AssemblyAI terminate send failed | stream_key=%s | error=%s',
                session.stream_key,
                exc,
            )
        finally:
            try:
                session.ws_app.close()
            except Exception:
                pass
            self._mark_closed(session)

    def _mark_closed(self, session: _AssemblyAiSession) -> None:
        with session.lock:
            if session.closed:
                return
            session.closed = True
        ASSEMBLYAI_SESSIONS_OPEN.dec()
        ASSEMBLYAI_SESSIONS_TERMINATED_TOTAL.inc()

    def _idle_cleanup_loop(self) -> None:
        while not self._idle_stop.wait(1.0):
            now_ms = int(time.time() * 1000)
            stale_keys: list[str] = []
            with self._lock:
                for key, session in self._sessions.items():
                    if (
                        session.last_audio_wall_ms
                        and now_ms - session.last_audio_wall_ms
                        > self._config.stream_idle_timeout_ms
                    ):
                        stale_keys.append(key)
            for key in stale_keys:
                logger.warning(
                    'AssemblyAI stream idle timeout; terminating to avoid billing leak | '
                    'stream_key=%s | idle_timeout_ms=%s',
                    key,
                    self._config.stream_idle_timeout_ms,
                )
                self.end_stream(key)

    def _get_session(self, stream_key: str) -> Optional[_AssemblyAiSession]:
        with self._lock:
            return self._sessions.get(stream_key)

    def _reset_turn_audio(self, session: _AssemblyAiSession) -> None:
        with session.lock:
            session.current_turn_pcm.clear()
            session.current_turn_start_ms = None

    @staticmethod
    def _extract_pcm(audio_data: bytes) -> bytes:
        if len(audio_data) >= WAV_HEADER_BYTES and audio_data[:4] == b'RIFF':
            return audio_data[WAV_HEADER_BYTES:]
        return audio_data

    @staticmethod
    def _duration_ms(pcm_data: bytes, sample_rate: int, channels: int) -> int:
        bytes_per_second = max(sample_rate * max(channels, 1) * 2, 1)
        return int((len(pcm_data) / bytes_per_second) * 1000)

    @staticmethod
    def _confidence_from_words(words: object) -> float:
        if not isinstance(words, list):
            return 0.0
        values: list[float] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            raw = word.get('confidence')
            if isinstance(raw, (int, float)):
                values.append(float(raw))
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _binary_opcode() -> int:
        import websocket

        return websocket.ABNF.OPCODE_BINARY
