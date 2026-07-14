"""Smoke test manual do DesktopWsGateway (não faz parte da suíte pytest).

Roda um gateway real em porta efêmera, conecta clientes de áudio e feedback
com um JWT HS256 e valida: ingestão de PCM coalescido, rejeição sem token e
broadcast de feedback.

Uso: python tests/smoke_ws_gateway.py
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt as pyjwt

from src.ws_gateway import DesktopWsAuthenticator, DesktopWsGateway, FeedbackHub
from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.types import TextAnalysisResult

SECRET = 'smoke-secret'
PORT = 18765


class FakeAudioService:
    def __init__(self):
        self.started = []
        self.chunks = []
        self.ended = []

    def start_stream(self, **kw):
        self.started.append(kw)

    def process_chunk(self, **kw):
        self.chunks.append(kw)

    def end_stream(self, **kw):
        self.ended.append(kw)


def make_token(tenant='tenant-1'):
    now = int(time.time())
    return pyjwt.encode(
        {
            'sub': 'user-1',
            'tid': tenant,
            'mid': 'm-1',
            'role': 'OWNER',
            'jti': 'j-1',
            'type': 'access',
            'iss': 'meet-backend',
            'aud': 'meet-platform',
            'iat': now,
            'exp': now + 300,
        },
        SECRET,
        algorithm='HS256',
    )


async def main():
    import websockets

    audio_service = FakeAudioService()
    hub = FeedbackHub()
    gateway = DesktopWsGateway(
        port=PORT,
        authenticator=DesktopWsAuthenticator(
            jwt_secret=SECRET,
            require_auth=True,
        ),
        feedback_hub=hub,
        audio_service=audio_service,
        coalesce_ms=40,
    )
    gateway.start()

    token = make_token()
    base = f'ws://127.0.0.1:{PORT}/ws'

    # 1) Sem token -> rejeitado
    try:
        async with websockets.connect(
            f'{base}?meetingId=m1&tenantId=tenant-1',
        ) as ws:
            await ws.recv()
        raise AssertionError('conexão sem token deveria ser rejeitada')
    except websockets.exceptions.ConnectionClosedError as exc:
        assert exc.rcvd and exc.rcvd.code == 1008, exc
        print('OK: conexão sem token rejeitada (1008)')

    # 2) Áudio: envia 10 frames de 640B (20ms @16k mono) -> coalesce 40ms
    audio_url = (
        f'{base}?meetingId=m1&participant=desktop&track=microphone'
        f'&sampleRate=16000&channels=1&tenantId=tenant-1'
        f'&participantRole=host&token={token}'
    )
    async with websockets.connect(audio_url) as ws:
        frame = bytes(640)
        for _ in range(10):
            await ws.send(frame)
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.3)
    await asyncio.sleep(0.3)
    assert audio_service.started, 'start_stream não foi chamado'
    assert audio_service.chunks, 'process_chunk não foi chamado'
    assert audio_service.ended, 'end_stream não foi chamado'
    total = sum(len(c['wav_data']) for c in audio_service.chunks)
    assert total == 6400, f'bytes processados: {total}'
    meta = audio_service.chunks[0]
    assert meta['tenant_id'] == 'tenant-1' and meta['participant_role'] == 'host'
    print(
        f'OK: áudio ingerido | chunks={len(audio_service.chunks)} | '
        f'bytes={total} | start/end ok',
    )

    # 3) Feedback: assina e recebe broadcast
    feedback_url = f'{base}?mode=feedback&meetingId=m1&tenantId=tenant-1&token={token}'
    async with websockets.connect(feedback_url) as ws:
        await asyncio.sleep(0.2)
        event = BackendFeedbackEvent(
            meeting_id='m1',
            participant_id='meet-remote',
            participant_name=None,
            participant_role='participant',
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=int(time.time() * 1000),
            window_start_ms=0,
            window_end_ms=int(time.time() * 1000),
            message='Text analysis ingress event',
            transcript_text='não vou comprar agora',
            transcript_confidence=0.92,
            analysis=TextAnalysisResult(
                direct_feedback='Cliente adiou a decisão — explore o motivo.',
                feedback_type='objection',
                confidence=0.85,
            ),
            tenant_id='tenant-1',
        )
        sent = hub.broadcast(event)
        assert sent, 'broadcast retornou False'
        raw = await asyncio.wait_for(ws.recv(), timeout=3)
        envelope = json.loads(raw)
        assert envelope['type'] == 'feedback'
        payload = envelope['payload']
        assert payload['message'] == 'Cliente adiou a decisão — explore o motivo.'
        assert payload['metadata']['feedbackType'] == 'objection'
        assert payload['tenantId'] == 'tenant-1'
        print(f"OK: feedback broadcast recebido | id={payload['id']}")

    # 4) direct_feedback vazio não é broadcast
    empty = BackendFeedbackEvent(
        meeting_id='m1',
        participant_id='meet-remote',
        participant_name=None,
        participant_role='participant',
        feedback_type='text_analysis_ingress',
        severity='info',
        ts_ms=int(time.time() * 1000),
        window_start_ms=0,
        window_end_ms=int(time.time() * 1000),
        message='',
        transcript_text='...',
        transcript_confidence=0.5,
        analysis=TextAnalysisResult(direct_feedback=''),
        tenant_id='tenant-1',
    )
    assert not hub.broadcast(empty)
    print('OK: evento sem direct_feedback não é broadcast')

    gateway.stop()
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    asyncio.run(main())
