"""Bounded post-publish Gemini specialist analysis for Live turns."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Deque, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ...metrics.realtime_metrics import (
    SPECIALIST_CALLS_TOTAL,
    SPECIALIST_DROPPED_TOTAL,
    SPECIALIST_ERRORS_TOTAL,
    SPECIALIST_LATENCY_MS,
    SPECIALIST_QUEUE_SIZE,
)
from .gemini_transport import GeminiTransportMode, generate_with_transport_chain
from .llm_state_validator import VALID_OBJECTION_CATEGORIES

logger = logging.getLogger(__name__)

_CONFIDENCE_WORDS: dict[str, float] = {
    'alta': 0.85,
    'alto': 0.85,
    'high': 0.85,
    'media': 0.6,
    'medio': 0.6,
    'medium': 0.6,
    'baixa': 0.35,
    'baixo': 0.35,
    'low': 0.35,
}


def _coerce_confidence(raw: object) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    text = str(raw).strip().lower().replace(',', '.')
    if text in _CONFIDENCE_WORDS:
        return _CONFIDENCE_WORDS[text]
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return 0.0


class SpecialistResult(BaseModel):
    """One combined SPIN, objection and compliance response."""

    source_turn_id: str
    fase_spin: Literal['neutro', 'situacao', 'problema', 'implicacao', 'necessidade'] = (
        'neutro'
    )
    proxima_pergunta_spin: str = Field(default='', max_length=500)
    alerta_risco_spin: bool = False
    objecoes_detectadas: list[str] = Field(default_factory=list)
    objection_hint: str = Field(default='', max_length=500)
    compliance_flagged: bool = False
    compliance_severity: Literal['info', 'warning', 'critical'] = 'info'
    compliance_reason: str = Field(default='', max_length=500)
    evidence_text: str = Field(default='', max_length=1000)
    secondary_feedback: str = Field(default='', max_length=500)
    secondary_feedback_type: Literal['risk', 'objection', 'clarification'] = (
        'clarification'
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode='before')
    @classmethod
    def coerce_llm_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get('compliance_severity') is None:
            data['compliance_severity'] = 'info'
        if data.get('compliance_reason') is None:
            data['compliance_reason'] = ''
        if data.get('secondary_feedback') is None:
            data['secondary_feedback'] = ''
        if data.get('evidence_text') is None:
            data['evidence_text'] = ''
        if data.get('objection_hint') is None:
            data['objection_hint'] = ''
        if data.get('proxima_pergunta_spin') is None:
            data['proxima_pergunta_spin'] = ''
        data['confidence'] = _coerce_confidence(data.get('confidence'))
        return data

    @field_validator('objecoes_detectadas')
    @classmethod
    def normalize_objections(cls, value: list[str]) -> list[str]:
        return [
            item
            for item in dict.fromkeys(str(raw).strip().lower() for raw in value)
            if item in VALID_OBJECTION_CATEGORIES
        ]

    def merged_state(self, current: dict[str, Any]) -> dict[str, Any]:
        merged = dict(current)
        merged['fase_spin'] = self.fase_spin
        merged['proxima_pergunta_spin'] = self.proxima_pergunta_spin
        merged['alerta_risco_spin'] = self.alerta_risco_spin
        existing = list(merged.get('objecoes_detectadas') or [])
        merged['objecoes_detectadas'] = list(
            dict.fromkeys([*existing, *self.objecoes_detectadas]),
        )
        return merged

    def next_turn_hint(self) -> str:
        parts = []
        if self.proxima_pergunta_spin:
            parts.append(f'Próxima pergunta SPIN sugerida: {self.proxima_pergunta_spin}')
        if self.objection_hint:
            parts.append(f'Objeção: {self.objection_hint}')
        if self.compliance_flagged and self.compliance_reason:
            parts.append(f'Compliance: {self.compliance_reason}')
        return ' | '.join(parts)[:1200]

    def metadata(self) -> dict[str, Any]:
        return {
            'spin': {
                'fase': self.fase_spin,
                'risco': self.alerta_risco_spin,
            },
            'objections': self.objecoes_detectadas,
            'compliance': {
                'flagged': self.compliance_flagged,
                'severity': self.compliance_severity,
                'reason': self.compliance_reason,
            },
        }


@dataclass(frozen=True)
class SpecialistSnapshot:
    tenant_id: str
    meeting_id: str
    participant_id: str
    participant_role: str
    turn_id: str
    speech_end_ms: int
    evidence_text: str
    primary_feedback: str
    conversation_state: dict[str, Any]
    host_context: str


class GeminiSpecialistAnalyzer:
    """Single generateContent call for all three specialist perspectives."""

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self._api_key = (api_key or '').strip()
        self._model_name = model_name
        self._cached_transport: Optional[GeminiTransportMode] = None

    def analyze(self, snapshot: SpecialistSnapshot) -> SpecialistResult:
        prompt = (
            'Analise um único turno de uma conversa de vendas em três perspectivas: '
            'SPIN, objeções e compliance. Use somente a evidência e o estado fornecidos. '
            'Não invente fatos. Retorne JSON puro com exatamente os campos: '
            'source_turn_id, fase_spin, proxima_pergunta_spin, alerta_risco_spin, '
            'objecoes_detectadas, objection_hint, compliance_flagged, '
            'compliance_severity, compliance_reason, evidence_text, '
            'secondary_feedback, secondary_feedback_type, confidence. '
            'confidence DEVE ser número entre 0.0 e 1.0 (nunca texto). '
            'compliance_severity DEVE ser info, warning ou critical (use info quando '
            'compliance_flagged=false). compliance_reason DEVE ser string (use "" se vazio). '
            'secondary_feedback deve ser curto e acionável; deixe vazio se não houver '
            'algo novo em relação ao feedback principal. Tipos permitidos: '
            'risk, objection, clarification.\n\n'
            f'source_turn_id={snapshot.turn_id}\n'
            f'evidencia={snapshot.evidence_text[:1000]}\n'
            f'feedback_principal={snapshot.primary_feedback[:500]}\n'
            f'contexto_host={snapshot.host_context[:1500]}\n'
            f'estado={json.dumps(snapshot.conversation_state, ensure_ascii=False)}'
        )
        response_text, transport = generate_with_transport_chain(
            api_key=self._api_key,
            model_name=self._model_name,
            prompt=prompt,
            preferred_mode=self._cached_transport,
        )
        self._cached_transport = transport
        result = SpecialistResult.model_validate_json(response_text)
        if result.source_turn_id != snapshot.turn_id:
            raise ValueError('Specialist source_turn_id mismatch')
        return result


SpecialistResultHook = Callable[[SpecialistSnapshot, SpecialistResult], None]
AnalyzeFn = Callable[[SpecialistSnapshot], SpecialistResult]


class LiveSpecialistRunner:
    """Latest-wins bounded queue isolated from the realtime publish path."""

    def __init__(
        self,
        analyze_fn: AnalyzeFn,
        on_result: SpecialistResultHook,
        *,
        max_queue_size: int = 32,
        timeout_ms: int = 8_000,
        max_age_ms: int = 120_000,
    ) -> None:
        self._analyze_fn = analyze_fn
        self._on_result = on_result
        self._max_queue_size = max_queue_size
        self._timeout_s = timeout_ms / 1000.0
        self._max_age_ms = max_age_ms
        self._pending: dict[str, SpecialistSnapshot] = {}
        self._order: Deque[str] = deque()
        self._latest_speech_end: dict[str, int] = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix='live-specialist-call',
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name='live-specialist-dispatcher',
            daemon=True,
        )
        self._worker.start()

    def enqueue(self, snapshot: SpecialistSnapshot) -> bool:
        with self._not_empty:
            if self._shutdown:
                return False
            self._latest_speech_end[snapshot.meeting_id] = max(
                snapshot.speech_end_ms,
                self._latest_speech_end.get(snapshot.meeting_id, 0),
            )
            if snapshot.meeting_id in self._pending:
                self._pending[snapshot.meeting_id] = snapshot
                SPECIALIST_DROPPED_TOTAL.labels(reason='latest_wins').inc()
                return True
            if len(self._order) >= self._max_queue_size:
                evicted = self._order.popleft()
                self._pending.pop(evicted, None)
                SPECIALIST_DROPPED_TOTAL.labels(reason='queue_full').inc()
            self._pending[snapshot.meeting_id] = snapshot
            self._order.append(snapshot.meeting_id)
            SPECIALIST_QUEUE_SIZE.set(len(self._order))
            self._not_empty.notify()
            return True

    def clear_meeting(self, meeting_id: str) -> None:
        with self._not_empty:
            self._pending.pop(meeting_id, None)
            self._order = deque(item for item in self._order if item != meeting_id)
            self._latest_speech_end.pop(meeting_id, None)
            SPECIALIST_QUEUE_SIZE.set(len(self._order))

    def shutdown(self, *, wait: bool = True) -> None:
        with self._not_empty:
            self._shutdown = True
            self._pending.clear()
            self._order.clear()
            SPECIALIST_QUEUE_SIZE.set(0)
            self._not_empty.notify_all()
        if wait:
            self._worker.join(timeout=max(1.0, self._timeout_s + 1.0))
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _worker_loop(self) -> None:
        while True:
            with self._not_empty:
                while not self._shutdown and not self._order:
                    self._not_empty.wait(timeout=0.5)
                if self._shutdown:
                    return
                meeting_id = self._order.popleft()
                snapshot = self._pending.pop(meeting_id)
                SPECIALIST_QUEUE_SIZE.set(len(self._order))
            self._process(snapshot)

    def _process(self, snapshot: SpecialistSnapshot) -> None:
        now_ms = int(time.time() * 1000)
        if now_ms - snapshot.speech_end_ms > self._max_age_ms:
            SPECIALIST_DROPPED_TOTAL.labels(reason='stale_job').inc()
            return
        SPECIALIST_CALLS_TOTAL.inc()
        started = time.perf_counter()
        future = self._executor.submit(self._analyze_fn, snapshot)
        try:
            result = future.result(timeout=self._timeout_s)
        except FutureTimeout:
            future.cancel()
            SPECIALIST_ERRORS_TOTAL.inc()
            SPECIALIST_DROPPED_TOTAL.labels(reason='timeout').inc()
            return
        except Exception:
            SPECIALIST_ERRORS_TOTAL.inc()
            logger.exception(
                'live.specialist.failed | meeting=%s | turnId=%s',
                snapshot.meeting_id,
                snapshot.turn_id,
            )
            return
        finally:
            SPECIALIST_LATENCY_MS.observe(
                (time.perf_counter() - started) * 1000.0,
            )

        with self._lock:
            latest = self._latest_speech_end.get(snapshot.meeting_id, 0)
        if snapshot.speech_end_ms < latest:
            SPECIALIST_DROPPED_TOTAL.labels(reason='stale_result').inc()
            return
        try:
            self._on_result(snapshot, result)
        except Exception:
            SPECIALIST_ERRORS_TOTAL.inc()
            logger.exception(
                'live.specialist.result_hook_failed | meeting=%s | turnId=%s',
                snapshot.meeting_id,
                snapshot.turn_id,
            )
