"""Google Gemini Live implementation of RealtimeCoachProvider."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from ..text_analysis.gemini_transport import uses_vertex_express_api_key
from .types import SessionContext

logger = logging.getLogger(__name__)

EMIT_FEEDBACK_TOOL = {
    'function_declarations': [
        {
            'name': 'emit_feedback',
            'description': (
                'Emit exactly one final coaching feedback for the just-finished '
                'customer speech turn. Call once per turnId; each new turnId needs a '
                'new call. Never reuse an old turnId. Never speak aloud — only call '
                'this tool.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'turnId': {
                        'type': 'string',
                        'description': 'Opaque turn id provided by the server for this utterance.',
                    },
                    'feedback': {
                        'type': 'string',
                        'description': 'Short actionable coaching tip for the seller, or empty if none.',
                    },
                    'confidence': {
                        'type': 'number',
                        'description': 'Confidence 0..1',
                    },
                    'feedback_type': {
                        'type': 'string',
                        'description': (
                            'objection|opportunity|rapport|closing|clarification|risk|null'
                        ),
                    },
                    'evidence_text': {
                        'type': 'string',
                        'description': 'Short literal quote from the customer speech.',
                    },
                    'estado': {
                        'type': 'object',
                        'description': 'Partial conversation state delta.',
                        'properties': {
                            'fase_spin': {
                                'type': 'string',
                                'description': 'neutro|situacao|problema|implicacao|necessidade',
                            },
                            'interesse': {
                                'type': 'string',
                                'description': 'baixo|medio|alto',
                            },
                            'engajamento': {
                                'type': 'string',
                                'description': 'baixo|medio|alto',
                            },
                            'resistencia': {
                                'type': 'string',
                                'description': 'baixa|media|alta',
                            },
                            'sentimento_cliente': {
                                'type': 'string',
                                'description': 'positivo|neutro|negativo',
                            },
                            'sentimento_tendencia': {
                                'type': 'string',
                                'description': 'subindo|estavel|caindo',
                            },
                            'objecoes_ativas': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': (
                                    'Open objection categories, e.g. preco, tempo, '
                                    'confianca, autoridade, necessidade.'
                                ),
                            },
                            'objecoes_resolvidas': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Objection categories already handled this call.',
                            },
                        },
                    },
                    'playbook_template_key': {
                        'type': 'string',
                    },
                    'playbook_variables': {
                        'type': 'object',
                    },
                },
                'required': [
                    'turnId',
                    'feedback',
                    'confidence',
                    'feedback_type',
                    'evidence_text',
                    'estado',
                ],
            },
        },
    ],
}

SYSTEM_INSTRUCTION = (
    'Você é um copiloto de vendas de baixa latência. Ouça o áudio do CLIENTE. '
    'Sempre que um turno de fala do cliente terminar, chame emit_feedback exatamente '
    'uma vez com o turnId daquele turno. Cada novo turnId é um turno novo — chame de '
    'novo. Nunca reutilize um turnId antigo. Nunca responda por voz. Se não houver '
    'feedback útil, chame emit_feedback com feedback="" e confidence=0. Priorize '
    'objection, opportunity, rapport, closing, clarification e risk. '
    'Em estado, preencha sentimento_cliente (positivo|neutro|negativo), '
    'sentimento_tendencia (subindo|estavel|caindo) e separe objecoes_ativas '
    'de objecoes_resolvidas usando as categorias preco/tempo/confianca/etc.'
)


class GeminiCoachSession:
    """Thin wrapper around google.genai AsyncSession."""

    def __init__(self, live: Any, types: Any, send_lock: Any) -> None:
        self._live = live
        self._types = types
        self._lock = send_lock

    async def send_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        async with self._lock:
            await self._live.send_realtime_input(
                audio=self._types.Blob(
                    data=pcm,
                    mime_type='audio/pcm;rate=16000',
                ),
            )

    async def send_activity(self, kind: Literal['start', 'end']) -> None:
        async with self._lock:
            if kind == 'start':
                await self._live.send_realtime_input(
                    activity_start=self._types.ActivityStart(),
                )
            else:
                await self._live.send_realtime_input(
                    activity_end=self._types.ActivityEnd(),
                )

    async def send_text(self, text: str) -> None:
        if not (text or '').strip():
            return
        async with self._lock:
            await self._live.send_realtime_input(text=text)

    def receive(self) -> AsyncIterator[Any]:
        return self._live.receive()

    async def ack_tools(self, function_calls: list[Any]) -> None:
        if not function_calls:
            return
        responses = []
        for fc in function_calls:
            name = getattr(fc, 'name', '') or ''
            responses.append(
                self._types.FunctionResponse(
                    id=getattr(fc, 'id', None) or str(uuid.uuid4()),
                    name=name or 'emit_feedback',
                    response={'result': 'ok'},
                ),
            )
        async with self._lock:
            await self._live.send_tool_response(function_responses=responses)

    @staticmethod
    def parse_tool_calls(response: Any) -> list[tuple[str, dict[str, Any], Any]]:
        tool_call = getattr(response, 'tool_call', None)
        if tool_call is None:
            return []
        out: list[tuple[str, dict[str, Any], Any]] = []
        for fc in getattr(tool_call, 'function_calls', None) or []:
            name = getattr(fc, 'name', '') or ''
            args = getattr(fc, 'args', None) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append((name, args, fc))
        return out


class GeminiLiveProvider:
    name = 'gemini'

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = 'gemini-3.1-flash-live-preview',
        context_window_tokens: int = 12_000,
    ) -> None:
        self._api_key = (api_key or '').strip()
        self._model_name = model_name
        self._context_window_tokens = context_window_tokens

    @asynccontextmanager
    async def open_session(self, ctx: SessionContext):
        from google import genai
        from google.genai import types
        import asyncio

        api_key = (ctx.api_key or self._api_key).strip()
        client = genai.Client(
            api_key=api_key,
            vertexai=uses_vertex_express_api_key(api_key),
        )
        config = self._build_config(types, ctx)
        send_lock = asyncio.Lock()
        async with client.aio.live.connect(
            model=self._model_name,
            config=config,
        ) as live:
            yield GeminiCoachSession(live, types, send_lock)

    def _build_config(self, types: Any, ctx: SessionContext) -> Any:
        thinking = None
        try:
            thinking = types.ThinkingConfig(thinking_level='minimal')
        except Exception:
            thinking = None

        system_instruction = SYSTEM_INSTRUCTION
        if ctx.catalog_prompt:
            system_instruction = f'{SYSTEM_INSTRUCTION}\n\n{ctx.catalog_prompt}'

        tokens = ctx.context_window_tokens or self._context_window_tokens
        kwargs: dict[str, Any] = {
            'response_modalities': ['AUDIO'],
            'system_instruction': system_instruction,
            'tools': [EMIT_FEEDBACK_TOOL],
            'realtime_input_config': {
                'automatic_activity_detection': {'disabled': True},
            },
            'context_window_compression': {
                'sliding_window': {},
                'trigger_tokens': tokens,
            },
        }
        if thinking is not None:
            kwargs['thinking_config'] = thinking
        if ctx.resumption_handle:
            kwargs['session_resumption'] = {'handle': ctx.resumption_handle}
        else:
            kwargs['session_resumption'] = {}
        try:
            return types.LiveConnectConfig(**kwargs)
        except Exception:
            return kwargs
