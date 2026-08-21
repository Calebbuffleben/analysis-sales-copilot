"""Post-publish specialist fan-out: never blocks the primary Live SLO."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional

from ...metrics.realtime_metrics import (
    SPECIALIST_CALLS_TOTAL,
    SPECIALIST_DROPPED_TOTAL,
    SPECIALIST_ERRORS_TOTAL,
    SPECIALIST_LATENCY_MS,
)
from ..text_analysis.gemini_transport import generate_with_transport_chain
from .catalog import SpecialistCatalog
from .json_util import parse_json as _parse_json
from .prompt import compile_specialist_prompt
from .types import SpecialistDef, SpecialistOutput, SpecialistTurnContext

logger = logging.getLogger(__name__)

ResultHook = Callable[[SpecialistTurnContext, list[SpecialistOutput]], None]


def _matches_trigger(spec: SpecialistDef, ctx: SpecialistTurnContext) -> bool:
    state = ctx.conversation_state or {}
    phase = str(state.get('fase_spin') or '').strip().lower()
    if spec.trigger_phases and phase and phase not in spec.trigger_phases:
        if phase != 'neutro':
            return False
    blob = ' '.join(
        [
            ctx.evidence_text,
            ctx.primary_feedback,
            json.dumps(state, ensure_ascii=False),
        ],
    ).lower()
    if spec.trigger_keywords and not any(k.lower() in blob for k in spec.trigger_keywords):
        if spec.trigger_phases:
            return True
        return False
    return True


class SpecialistFanout:
    """Router + parallel gather + merge. Fail-open per specialist."""

    def __init__(
        self,
        catalog: SpecialistCatalog,
        *,
        api_key: str,
        on_result: Optional[ResultHook] = None,
        enabled: bool = True,
    ) -> None:
        self._catalog = catalog
        self._api_key = (api_key or '').strip()
        self._on_result = on_result
        self._enabled = enabled
        self._last_fired: dict[str, dict[str, int]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def enqueue(self, ctx: SpecialistTurnContext) -> bool:
        if not self._enabled or not self._api_key:
            return False
        loop = self._loop
        if loop is None or not loop.is_running():
            SPECIALIST_DROPPED_TOTAL.labels(reason='no_loop').inc()
            return False
        asyncio.run_coroutine_threadsafe(self._run(ctx), loop)
        return True

    async def _run(self, ctx: SpecialistTurnContext) -> None:
        specs = [
            spec
            for spec in self._catalog.enabled_for(ctx.selected_keys)
            if _matches_trigger(spec, ctx) and self._cooldown_ok(spec, ctx)
        ]
        if not specs:
            return
        tasks = [self._call_one(spec, ctx) for spec in specs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[SpecialistOutput] = []
        for spec, result in zip(specs, results):
            if isinstance(result, SpecialistOutput):
                outputs.append(result)
                self._mark_fired(spec, ctx)
            elif isinstance(result, Exception):
                SPECIALIST_ERRORS_TOTAL.inc()
                logger.warning(
                    'specialist.failed | key=%s | meeting=%s | error=%s',
                    spec.key,
                    ctx.meeting_id,
                    result,
                )
        outputs.sort(key=lambda item: (-item.confidence, item.key))
        if self._on_result and outputs:
            try:
                self._on_result(ctx, outputs)
            except Exception:
                logger.exception('specialist.result_hook_failed')

    async def _call_one(
        self,
        spec: SpecialistDef,
        ctx: SpecialistTurnContext,
    ) -> SpecialistOutput:
        SPECIALIST_CALLS_TOTAL.inc()
        started = time.perf_counter()
        prompt = compile_specialist_prompt(spec, ctx)
        timeout = max(0.2, spec.max_latency_ms / 1000.0)
        try:
            text, _transport = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_with_transport_chain,
                    api_key=self._api_key,
                    model_name=spec.model,
                    prompt=prompt,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            SPECIALIST_DROPPED_TOTAL.labels(reason='timeout').inc()
            SPECIALIST_LATENCY_MS.observe((time.perf_counter() - started) * 1000.0)
            raise TimeoutError(f'{spec.key} exceeded {spec.max_latency_ms}ms')
        SPECIALIST_LATENCY_MS.observe((time.perf_counter() - started) * 1000.0)
        payload = _parse_json(text)
        feedback = str(payload.get('secondary_feedback') or '').strip()
        return SpecialistOutput(
            key=spec.key,
            name=spec.name,
            secondary_feedback=feedback,
            secondary_feedback_type=str(
                payload.get('secondary_feedback_type') or 'clarification',
            ),
            confidence=float(payload.get('confidence') or 0.0),
            evidence_text=str(payload.get('evidence_text') or ctx.evidence_text),
            next_turn_hint=str(payload.get('next_turn_hint') or payload.get('proxima_pergunta_spin') or ''),
            metadata={
                'specialist': spec.key,
                'name': spec.name,
                'source': spec.source,
                'raw': {
                    k: payload.get(k)
                    for k in (
                        'fase_spin',
                        'alerta_risco_spin',
                        'objecoes_detectadas',
                        'compliance_flagged',
                    )
                    if k in payload
                },
            },
        )

    def _cooldown_ok(self, spec: SpecialistDef, ctx: SpecialistTurnContext) -> bool:
        last = self._last_fired.get(ctx.meeting_id, {}).get(spec.key, 0)
        return ctx.speech_end_ms - last >= spec.cooldown_sec * 1000

    def _mark_fired(self, spec: SpecialistDef, ctx: SpecialistTurnContext) -> None:
        bucket = self._last_fired.setdefault(ctx.meeting_id, {})
        bucket[spec.key] = ctx.speech_end_ms

    def clear_meeting(self, meeting_id: str) -> None:
        self._last_fired.pop(meeting_id, None)
