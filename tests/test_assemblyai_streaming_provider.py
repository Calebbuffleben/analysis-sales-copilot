from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace
from typing import Any

from src.modules.transcription.assemblyai_streaming_provider import (
    AssemblyAiStreamConfig,
    AssemblyAiStreamingProvider,
)


class _FakeWebSocketApp:
    instances: list['_FakeWebSocketApp'] = []

    def __init__(
        self,
        url: str,
        header: dict[str, str],
        on_open: Any,
        on_message: Any,
        on_error: Any,
        on_close: Any,
    ) -> None:
        self.url = url
        self.header = header
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent: list[tuple[Any, Any]] = []
        self.closed = False
        _FakeWebSocketApp.instances.append(self)

    def run_forever(self) -> None:
        self.on_open(self)

    def send(self, payload: Any, opcode: Any = None) -> None:
        self.sent.append((payload, opcode))
        if isinstance(payload, str) and json.loads(payload).get('type') == 'Terminate':
            self.on_message(
                self,
                json.dumps(
                    {
                        'type': 'Termination',
                        'audio_duration_seconds': 1.0,
                        'session_duration_seconds': 1.1,
                    },
                ),
            )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.on_close(self, 1000, 'closed')


def _install_fake_websocket(monkeypatch: Any) -> None:
    _FakeWebSocketApp.instances.clear()
    fake_module = SimpleNamespace(
        WebSocketApp=_FakeWebSocketApp,
        ABNF=SimpleNamespace(OPCODE_BINARY=2),
    )
    monkeypatch.setitem(sys.modules, 'websocket', fake_module)


def test_final_turn_maps_to_transcription_chunk(monkeypatch: Any) -> None:
    _install_fake_websocket(monkeypatch)
    received: list[tuple[str, Any, dict[str, object]]] = []
    callback_event = threading.Event()

    def on_final(stream_key: str, chunk: Any, stats: dict[str, object]) -> None:
        received.append((stream_key, chunk, stats))
        callback_event.set()

    provider = AssemblyAiStreamingProvider(
        AssemblyAiStreamConfig(
            api_key='test-key',
            connect_timeout_seconds=0.2,
            termination_timeout_seconds=0.0,
        ),
        on_final,
    )

    meta = {
        'meeting_id': 'meet-1',
        'participant_id': 'seller',
        'track': 'mic',
        'sample_rate': 16000,
        'channels': 1,
        'timestamp_ms': 10_000,
        'tenant_id': 'tenant-1',
    }
    provider.send_audio('meet-1:seller:mic', b'\x01\x00' * 1600, meta)
    provider._on_message(
        'meet-1:seller:mic',
        json.dumps(
            {
                'type': 'Turn',
                'turn_order': 1,
                'end_of_turn': True,
                'turn_is_formatted': True,
                'transcript': 'Cliente confirmou o problema.',
                'words': [
                    {'text': 'Cliente', 'confidence': 0.9},
                    {'text': 'confirmou', 'confidence': 0.8},
                ],
            },
        ),
    )

    assert callback_event.wait(timeout=1.0)
    stream_key, chunk, stats = received[0]
    assert stream_key == 'meet-1:seller:mic'
    assert chunk.meeting_id == 'meet-1'
    assert chunk.participant_id == 'seller'
    assert chunk.track == 'mic'
    assert chunk.tenant_id == 'tenant-1'
    assert chunk.text == 'Cliente confirmou o problema.'
    assert abs(chunk.confidence - 0.85) < 0.0001
    assert chunk.window_end_ms == 10_000
    assert stats['samples_count'] == 1600

    provider.close_all()


def test_partial_turn_does_not_emit_transcript(monkeypatch: Any) -> None:
    _install_fake_websocket(monkeypatch)
    received: list[object] = []
    provider = AssemblyAiStreamingProvider(
        AssemblyAiStreamConfig(
            api_key='test-key',
            connect_timeout_seconds=0.2,
            termination_timeout_seconds=0.0,
        ),
        lambda *_args: received.append(_args),
    )

    meta = {
        'meeting_id': 'meet-1',
        'participant_id': 'seller',
        'track': 'mic',
        'sample_rate': 16000,
        'channels': 1,
        'timestamp_ms': 10_000,
        'tenant_id': 'tenant-1',
    }
    provider.send_audio('meet-1:seller:mic', b'\x01\x00' * 1600, meta)
    provider._on_message(
        'meet-1:seller:mic',
        json.dumps(
            {
                'type': 'Turn',
                'end_of_turn': False,
                'transcript': 'Cliente confirmou',
            },
        ),
    )

    assert received == []
    provider.close_all()


def test_end_stream_sends_terminate_message(monkeypatch: Any) -> None:
    _install_fake_websocket(monkeypatch)
    provider = AssemblyAiStreamingProvider(
        AssemblyAiStreamConfig(
            api_key='test-key',
            connect_timeout_seconds=0.2,
            termination_timeout_seconds=0.0,
        ),
        lambda *_args: None,
    )
    provider.send_audio(
        'meet-1:seller:mic',
        b'\x01\x00' * 1600,
        {
            'meeting_id': 'meet-1',
            'participant_id': 'seller',
            'track': 'mic',
            'sample_rate': 16000,
            'channels': 1,
            'timestamp_ms': 10_000,
            'tenant_id': 'tenant-1',
        },
    )
    fake_ws = _FakeWebSocketApp.instances[0]

    provider.end_stream('meet-1:seller:mic')

    assert any(
        isinstance(payload, str) and json.loads(payload).get('type') == 'Terminate'
        for payload, _opcode in fake_ws.sent
    )
    provider.close_all()
