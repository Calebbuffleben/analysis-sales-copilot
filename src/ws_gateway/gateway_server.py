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
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from .feedback_hub import FeedbackHub
from .jwt_auth import DesktopWsAuthenticator, WsAuthError
from ..modules.acoustic_fingerprint.pcm_v2 import (
    AcousticLabelBuffer,
    is_pcm_v2,
    parse_label_control,
    try_decode_pcm_v2,
)

if TYPE_CHECKING:
    from ..services.audio_service import AudioService
    from ..modules.text_analysis.gemini_live_session import GeminiLiveManager
    from ..modules.playbooks.catalog_cache import PlaybookCatalogCache

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
        'seller_room_id',
        'pcm_version',
        'label_buffer',
        'buffer',
        'buffer_started_ms',
        'sequence',
        'started',
        'acoustic_class',
        'matched_seller_id',
        'correlation_confidence',
        'selected_specialists',
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
        seller_room_id: str = '',
        pcm_version: int = 1,
        selected_specialists: tuple[str, ...] = (),
    ) -> None:
        self.meeting_id = meeting_id
        self.participant_id = participant_id
        self.track = track
        self.sample_rate = sample_rate
        self.channels = channels
        self.tenant_id = tenant_id
        self.participant_role = participant_role
        self.seller_room_id = seller_room_id
        self.pcm_version = pcm_version
        self.selected_specialists = selected_specialists
        self.label_buffer = AcousticLabelBuffer()
        self.buffer = bytearray()
        self.buffer_started_ms: Optional[int] = None
        self.sequence = 0
        self.started = False
        self.acoustic_class = 'unknown'
        self.matched_seller_id = ''
        self.correlation_confidence = 0.0


class DesktopWsGateway:
    """Runs the websockets server on a dedicated thread with its own loop."""

    def __init__(
        self,
        *,
        port: int,
        authenticator: DesktopWsAuthenticator,
        feedback_hub: FeedbackHub,
        audio_service: 'AudioService',
        coalesce_ms: int = 40,
        host: str = '0.0.0.0',
        live_manager: Optional['GeminiLiveManager'] = None,
        catalog_cache: Optional['PlaybookCatalogCache'] = None,
        lifecycle_reporter: Optional[Callable[..., None]] = None,
    ) -> None:
        self._port = port
        self._host = host
        self._authenticator = authenticator
        self._feedback_hub = feedback_hub
        self._audio_service = audio_service
        self._coalesce_ms = max(20, coalesce_ms)
        self._live_manager = live_manager
        self._catalog_cache = catalog_cache
        self._lifecycle_reporter = lifecycle_reporter
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._started = threading.Event()
        self._connections: set[Any] = set()

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
            asyncio.run_coroutine_threadsafe(self._drain_and_stop(), loop)
        if self._thread is not None:
            self._thread.join(timeout=8)

    async def _drain_and_stop(self) -> None:
        payload = json.dumps({'type': 'session-migrate'})
        for connection in list(self._connections):
            try:
                await connection.send(payload)
            except Exception:
                pass
        if self._stop_event is not None:
            self._stop_event.set()

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

    async def _process_request(self, *args: Any) -> Any:
        """HTTP GET /health for Cloud Run probes (same PORT as WSS).

        Supports websockets 12 ``(path, headers)`` and 13+ ``(connection, request)``.
        """
        from .health_handshake import health_http_result

        return health_http_result(*args)

    async def _serve(self) -> None:
        import websockets

        self._stop_event = asyncio.Event()
        async with websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=2 ** 20,
            process_request=self._process_request,
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

        self._connections.add(websocket)
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
            self._connections.discard(websocket)
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
        seller_room_id = _sanitize(query.get('sellerRoomId') or '')
        try:
            pcm_version = int(query.get('pcmVersion') or 1)
        except ValueError:
            pcm_version = 1
        try:
            sample_rate = int(query.get('sampleRate') or 16000)
        except ValueError:
            sample_rate = 16000
        try:
            channels = max(1, int(query.get('channels') or 1))
        except ValueError:
            channels = 1
        selected_specialists = tuple(
            _sanitize(item)
            for item in (query.get('specialists') or '').split(',')
            if item.strip()
        )

        state = _AudioStreamState(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
            tenant_id=tenant_id,
            participant_role=participant_role,
            seller_room_id=seller_room_id,
            pcm_version=pcm_version,
            selected_specialists=selected_specialists,
        )
        flush_threshold_bytes = max(
            int(sample_rate * channels * 2 * 0.05),
            int(sample_rate * channels * 2 * (self._coalesce_ms / 1000.0)),
        )
        process_lock = asyncio.Lock()
        if self._catalog_cache is not None and tenant_id:
            asyncio.create_task(asyncio.to_thread(self._catalog_cache.warm, tenant_id))
        if self._live_manager is not None and participant_role != 'host':
            asyncio.create_task(
                asyncio.to_thread(
                    self._live_manager.warm_session,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    sample_rate=sample_rate,
                    channels=channels,
                    selected_specialists=selected_specialists,
                ),
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
        self._report_lifecycle(
            event='open',
            tenant_id=tenant_id,
            meeting_id=meeting_id,
            user_id=user_id,
            participant_id=participant_id,
            participant_role=participant_role,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
        )
        opened_ms = int(time.time() * 1000)

        try:
            async for frame in websocket:
                if isinstance(frame, (bytes, bytearray, memoryview)):
                    await self._on_audio_bytes(
                        state,
                        bytes(frame),
                        flush_threshold_bytes,
                        process_lock,
                    )
                else:
                    # Text frames: ping or acoustic_label control.
                    self._on_audio_control(state, websocket, frame)
        except Exception:
            logger.debug('ws audio connection closed with error', exc_info=True)
        finally:
            await asyncio.to_thread(self._finish_audio_stream, state)
            self._report_lifecycle(
                event='close',
                tenant_id=tenant_id,
                meeting_id=meeting_id,
                user_id=user_id,
                participant_id=participant_id,
                participant_role=participant_role,
                track=track,
                sample_rate=sample_rate,
                channels=channels,
                duration_ms=max(0, int(time.time() * 1000) - opened_ms),
                chunks_received=state.sequence,
            )

    def _on_control_message(
        self,
        websocket: Any,
        raw: str,
        state: Optional[_AudioStreamState] = None,
    ) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(message, dict):
            return
        if message.get('type') == 'set-specialists' and state is not None:
            raw_keys = message.get('specialists') or message.get('keys') or []
            if isinstance(raw_keys, str):
                keys = tuple(_sanitize(item) for item in raw_keys.split(',') if item.strip())
            elif isinstance(raw_keys, list):
                keys = tuple(_sanitize(str(item)) for item in raw_keys if str(item).strip())
            else:
                keys = ()
            state.selected_specialists = keys
            if self._live_manager is not None:
                self._live_manager.set_selected_specialists(state.meeting_id, keys)
            return
        if message.get('type') == 'ping' and websocket is not None:
            asyncio.get_running_loop().create_task(
                websocket.send(
                    json.dumps({'type': 'pong', 'ts': int(time.time() * 1000)}),
                ),
            )

    def _on_audio_control(
        self,
        state: _AudioStreamState,
        websocket: Any,
        raw: str,
    ) -> None:
        label = parse_label_control(raw)
        if label is None:
            self._on_control_message(websocket, raw, state)
            return
        state.label_buffer.upsert(label)
        state.acoustic_class = label.acoustic_class
        state.matched_seller_id = label.matched_seller_id or ''
        state.correlation_confidence = label.confidence

    # ------------------------------------------------------------------ #
    # Audio plumbing (runs on the gateway loop; heavy work offloaded)
    # ------------------------------------------------------------------ #

    def _report_lifecycle(self, **payload: Any) -> None:
        reporter = self._lifecycle_reporter
        if reporter is None:
            return
        try:
            reporter(**payload)
        except Exception:
            logger.exception('session lifecycle report failed')

    async def _on_audio_bytes(
        self,
        state: _AudioStreamState,
        data: bytes,
        flush_threshold_bytes: int,
        process_lock: asyncio.Lock,
    ) -> None:
        now_ms = int(time.time() * 1000)
        pcm = data
        if is_pcm_v2(data):
            framed = try_decode_pcm_v2(data)
            if framed is None:
                return
            pcm = framed.pcm
            label = state.label_buffer.resolve_for_label_id(framed.label_id)
            if label is not None:
                state.acoustic_class = label.acoustic_class
                state.matched_seller_id = label.matched_seller_id or ''
                state.correlation_confidence = label.confidence

        if not state.buffer:
            state.buffer_started_ms = now_ms
        state.buffer.extend(pcm)

        elapsed_ms = now_ms - (state.buffer_started_ms or now_ms)
        if (
            len(state.buffer) >= flush_threshold_bytes
            or elapsed_ms >= self._coalesce_ms
        ):
            flush_pcm = bytes(state.buffer)
            state.buffer.clear()
            state.buffer_started_ms = None
            state.sequence += 1
            sequence = state.sequence
            async with process_lock:
                await asyncio.to_thread(self._process_pcm, state, flush_pcm, sequence, now_ms)

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
                acoustic_class=state.acoustic_class,
                seller_room_id=state.seller_room_id,
                matched_seller_id=state.matched_seller_id,
                correlation_confidence=state.correlation_confidence,
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
