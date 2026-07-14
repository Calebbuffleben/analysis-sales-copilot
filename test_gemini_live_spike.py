#!/usr/bin/env python3
"""Isolated Gemini Live spike (manual).

Validates Free/Paid key can open a Live session, send PCM activity, and receive
an emit_feedback tool call. Not part of the default pytest suite.

Usage (inside python-service container or venv):
  python test_gemini_live_spike.py
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')


def _keys() -> list[str]:
    multi = os.getenv('GEMINI_API_KEYS') or ''
    cleaned = multi.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1]
    keys = [k.strip().strip('"').strip("'") for k in cleaned.split(',') if k.strip()]
    if keys:
        return keys
    single = (os.getenv('GEMINI_API_KEY') or '').strip().strip('"').strip("'")
    return [single] if single else []


def _tone(ms: int = 800, sample_rate: int = 16000, amplitude: int = 6000) -> bytes:
    n = int(sample_rate * ms / 1000)
    return b''.join(struct.pack('<h', amplitude if (i // 40) % 2 else -amplitude) for i in range(n))


async def run_spike(api_key: str, model: str) -> dict:
    from google import genai
    from google.genai import types

    from src.modules.text_analysis.gemini_live_session import (
        EMIT_FEEDBACK_TOOL,
        SYSTEM_INSTRUCTION,
    )

    client = genai.Client(api_key=api_key, vertexai=api_key.startswith('AQ.'))
    config = {
        'response_modalities': ['AUDIO'],
        'system_instruction': SYSTEM_INSTRUCTION,
        'tools': [EMIT_FEEDBACK_TOOL],
        'realtime_input_config': {
            'automatic_activity_detection': {'disabled': True},
        },
        'thinking_config': types.ThinkingConfig(thinking_level='minimal'),
    }

    result = {
        'model': model,
        'tool_call': False,
        'vad_end_to_tool_ms': None,
        'unexpected_audio_bytes': 0,
        'error': None,
    }

    turn_id = f'spike-{int(time.time())}'
    speech_end_ms = None

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(
                text=(
                    f'Turno iniciado. turnId="{turn_id}". '
                    'Ao final chame emit_feedback uma vez com este turnId. '
                    'O cliente disse: "está muito caro para mim agora".'
                ),
            )
            pcm = _tone(900)
            await session.send_realtime_input(
                audio=types.Blob(data=pcm, mime_type='audio/pcm;rate=16000'),
            )
            speech_end_ms = int(time.time() * 1000)
            await session.send_realtime_input(activity_end=types.ActivityEnd())

            deadline = time.time() + 12.0
            async for response in session.receive():
                if getattr(response, 'data', None):
                    result['unexpected_audio_bytes'] += len(response.data)
                tool_call = getattr(response, 'tool_call', None)
                if tool_call is not None:
                    result['tool_call'] = True
                    result['vad_end_to_tool_ms'] = max(
                        0,
                        int(time.time() * 1000) - int(speech_end_ms or 0),
                    )
                    for fc in tool_call.function_calls or []:
                        args = getattr(fc, 'args', {}) or {}
                        result['args'] = args if isinstance(args, dict) else str(args)
                        await session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={'result': 'ok'},
                                ),
                            ],
                        )
                    break
                if time.time() > deadline:
                    result['error'] = 'timeout waiting for tool_call'
                    break
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def main() -> int:
    keys = _keys()
    if not keys:
        print('❌ FAIL: set GEMINI_API_KEY or GEMINI_API_KEYS')
        return 1
    model = (
        os.getenv('LIVE_MODEL') or 'gemini-3.1-flash-live-preview'
    ).strip()
    print(f'Live spike | model={model} | key_prefix={keys[0][:8]}...')
    result = asyncio.run(run_spike(keys[0], model))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('error'):
        print('❌ FAIL')
        return 2
    if not result.get('tool_call'):
        print('❌ FAIL: no emit_feedback tool call')
        return 3
    latency = result.get('vad_end_to_tool_ms')
    if latency is not None and latency > 850:
        print(f'⚠️  tool call ok but latency {latency}ms > 850ms budget')
    else:
        print('✅ PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
