"""WebSocket gateway for direct desktop <-> python-service realtime traffic.

One endpoint, two connection modes selected by the ``mode`` query param:

- ``mode=audio`` (default): binary frames are PCM s16le audio, coalesced
  locally (~100ms) and fed straight into ``AudioService`` — same contract
  the backend gRPC ``StreamAudio`` path uses. The connection also receives
  feedback JSON frames (bidirectional channel).
- ``mode=feedback``: receive-only subscription to feedback events for a
  ``(tenantId, meetingId)`` room.

Query params mirror the backend ``/egress-audio`` URL so the desktop can
reuse its URL builder: ``meetingId``, ``participant``, ``track``,
``sampleRate``, ``channels``, ``token``, ``tenantId``, ``participantRole``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import parse_qs, urlparse

from .feedback_hub import FeedbackHub
from .jwt_auth import DesktopWsAuthenticator, WsAuthError

if TYPE_CHECKING:
    from ..services.audio_service import AudioService

logger = logging.getLogger(__name__)

_CLOSE_POLICY_VIOLATION = 1008


def _sanitize(value: str) -> str:
    import re

    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', str(value or ''))[:128]


class _AudioStreamState:
    """Per-connection audio coalescing buffer + stream identity."""

    __slots__ = (
        'meeting_id',
        'participant_id',
        'track',
        'sample_rate',
        'channels',
        'tenant_id',
        'participant_role',
        'buffer',
        'buffer_started_ms',
        'sequence',
        'started',
    )

    def __init__(
        self,
        *,
        meeting_id: str,
        participant_id: str,
        track: str,
        sample_rate: int,
        channels: int,
        tenant_id: str,
        participant_role: str,
    ) -> None:
        self.meeting_id = meeting_id
        self.participant_id = participant_id
        self.track = track
        self.sample_rate = sample_rate
        self.channels = channels
        self.tenant_id = tenant_id
        self.participant_role = participant_role
        self.buffer = bytearray()
        self.buffer_started_ms: Optional[int] = None
        self.sequence = 0
        self.started = False


class DesktopWsGateway:
    """Runs the websockets server on a dedicated thread with its own loop."""

    def __init__(
        self,
        *,
        port: int,
        authenticator: DesktopWsAuthenticator,
        feedback_hub: FeedbackHub,
        audio_service: 'AudioService',
        coalesce_ms: int = 100,
        host: str = '0.0.0.0',
    ) -> None:
        self._port = port
        self._host = host
        self._authenticator = authenticator
        self._feedback_hub = feedback_hub
        self._audio_service = audio_service
        self._coalesce_ms = max(20, coalesce_ms)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._started = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name='desktop-ws-gateway',
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise RuntimeError('Desktop WS gateway failed to start within 10s')
        logger.info(
            '🔌 Desktop WS gateway listening | ws://%s:%s | coalesce_ms=%s',
            self._host,
            self._port,
            self._coalesce_ms,
        )

    def stop(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        if loop is None or stop_event is None:
            return
        if not loop.is_closed():
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._feedback_hub.attach_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception:
            logger.exception('Desktop WS gateway crashed')
        finally:
            loop.close()

    async def _serve(self) -> None:
        import websockets

        self._stop_event = asyncio.Event()
        async with websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=2 ** 20,
        ):
            self._started.set()
            await self._stop_event.wait()

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #

    @staticmethod
    def _request_path(websocket: Any) -> str:
        request = getattr(websocket, 'request', None)
        if request is not None and getattr(request, 'path', None):
            return str(request.path)
        return str(getattr(websocket, 'path', '') or '')

    async def _handle_connection(self, websocket: Any) -> None:
        path = self._request_path(websocket)
        query = {
            key: values[0]
            for key, values in parse_qs(urlparse(path).query).items()
            if values
        }

        mode = (query.get('mode') or 'audio').strip().lower()
        meeting_id = _sanitize(query.get('meetingId') or query.get('room') or '')
        tenant_hint = _sanitize(query.get('tenantId') or '')
        token = query.get('token')

        if not meeting_id:
            await websocket.close(_CLOSE_POLICY_VIOLATION, 'missing meetingId')
            return

        try:
            auth = self._authenticator.authenticate(token, tenant_hint or None)
        except WsAuthError as exc:
            logger.warning(
                'ws connection rejected | mode=%s | meetingId=%s | reason=%s',
                mode,
                meeting_id,
                exc,
            )
            await websocket.close(_CLOSE_POLICY_VIOLATION, f'unauthorized: {exc}')
            return

        tenant_id = auth.tenant_id or tenant_hint

        self._feedback_hub.register(tenant_id, meeting_id, websocket)
        try:
            if mode == 'feedback':
                await self._run_feedback_connection(websocket)
            else:
                await self._run_audio_connection(
                    websocket,
                    query,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    user_id=auth.user_id,
                )
        finally:
            self._feedback_hub.unregister(tenant_id, meeting_id, websocket)

    async def _run_feedback_connection(self, websocket: Any) -> None:
        # Receive-only subscription: drain (and ignore) inbound frames so
        # pings/pongs and closes are processed until the client disconnects.
        try:
            async for _ in websocket:
                pass
        except Exception:
            logger.debug('ws feedback connection closed with error', exc_info=True)

    async def _run_audio_connection(
        self,
        websocket: Any,
        query: dict[str, str],
        *,
        meeting_id: str,
        tenant_id: str,
        user_id: str,
    ) -> None:
        participant_id = _sanitize(query.get('participant') or user_id or 'desktop')
        track = _sanitize(query.get('track') or 'tab-audio')
        participant_role = _sanitize(query.get('participantRole') or '')
        try:
            sample_rate = int(query.get('sampleRate') or 16000)
        except ValueError:
            sample_rate = 16000
        try:
            channels = max(1, int(query.get('channels') or 1))
        except ValueError:
            channels = 1

        state = _AudioStreamState(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
            tenant_id=tenant_id,
            participant_role=participant_role,
        )
        flush_threshold_bytes = max(
            int(sample_rate * channels * 2 * 0.05),
            int(sample_rate * channels * 2 * (self._coalesce_ms / 1000.0)),
        )
        # Single worker per connection: preserves chunk ordering for STT
        # while keeping process_chunk off the event loop.
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f'ws-audio-{participant_id[:16]}',
        )

        logger.info(
            '🎙️ ws audio ingress started | meetingId=%s | participantId=%s | '
            'track=%s | sampleRate=%s | channels=%s | flushBytes=%s',
            meeting_id,
            participant_id,
            track,
            sample_rate,
            channels,
            flush_threshold_bytes,
        )

        try:
            async for frame in websocket:
                if isinstance(frame, (bytes, bytearray, memoryview)):
                    self._on_audio_bytes(
                        state,
                        bytes(frame),
                        flush_threshold_bytes,
                        executor,
                    )
                else:
                    # Text frames on the audio channel are control messages;
                    # only `ping` is understood today.
                    self._on_control_message(websocket, frame)
        except Exception:
            logger.debug('ws audio connection closed with error', exc_info=True)
        finally:
            # end_stream runs on the same single worker, AFTER pending chunks.
            executor.submit(self._finish_audio_stream, state)
            executor.shutdown(wait=False)

    def _on_control_message(self, websocket: Any, raw: str) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if isinstance(message, dict) and message.get('type') == 'ping':
            # Called from within the gateway loop (async for), so a task is enough.
            asyncio.get_running_loop().create_task(
                websocket.send(
                    json.dumps({'type': 'pong', 'ts': int(time.time() * 1000)}),
                ),
            )

    # ------------------------------------------------------------------ #
    # Audio plumbing (runs on the gateway loop; heavy work offloaded)
    # ------------------------------------------------------------------ #

    def _on_audio_bytes(
        self,
        state: _AudioStreamState,
        data: bytes,
        flush_threshold_bytes: int,
        executor: ThreadPoolExecutor,
    ) -> None:
        now_ms = int(time.time() * 1000)
        if not state.buffer:
            state.buffer_started_ms = now_ms
        state.buffer.extend(data)

        elapsed_ms = now_ms - (state.buffer_started_ms or now_ms)
        if (
            len(state.buffer) >= flush_threshold_bytes
            or elapsed_ms >= self._coalesce_ms * 2
        ):
            pcm = bytes(state.buffer)
            state.buffer.clear()
            state.buffer_started_ms = None
            state.sequence += 1
            sequence = state.sequence
            # process_chunk fans out to AssemblyAI/buffers; keep it off the
            # event loop so slow consumers never stall audio reads.
            executor.submit(self._process_pcm, state, pcm, sequence, now_ms)

    def _process_pcm(
        self,
        state: _AudioStreamState,
        pcm: bytes,
        sequence: int,
        timestamp_ms: int,
    ) -> None:
        try:
            if not state.started:
                state.started = True
                self._audio_service.start_stream(
                    meeting_id=state.meeting_id,
                    participant_id=state.participant_id,
                    track=state.track,
                    sample_rate=state.sample_rate,
                    channels=state.channels,
                )
            self._audio_service.process_chunk(
                meeting_id=state.meeting_id,
                participant_id=state.participant_id,
                track=state.track,
                wav_data=pcm,
                sequence=sequence,
                timestamp_ms=timestamp_ms,
                tenant_id=state.tenant_id,
                participant_role=state.participant_role,
            )
        except Exception:
            logger.exception(
                'ws audio chunk processing failed | meetingId=%s | participantId=%s',
                state.meeting_id,
                state.participant_id,
            )

    def _finish_audio_stream(self, state: _AudioStreamState) -> None:
        if not state.started:
            return
        try:
            self._audio_service.end_stream(
                meeting_id=state.meeting_id,
                participant_id=state.participant_id,
                track=state.track,
            )
            logger.info(
                '✅ ws audio ingress finished | meetingId=%s | participantId=%s | chunks=%s',
                state.meeting_id,
                state.participant_id,
                state.sequence,
            )
        except Exception:
            logger.exception('ws audio end_stream failed')
