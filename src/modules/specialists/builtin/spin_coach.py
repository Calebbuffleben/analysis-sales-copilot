"""SPIN + objection + compliance builtin (replaces the hardcoded live_specialist prompt)."""

from __future__ import annotations

import json

from ..decorator import specialist
from ..types import SpecialistDef, SpecialistTurnContext


@specialist(
    key='spin_coach',
    name='Coach SPIN',
    description='Observa fase SPIN, objeções e risco de compliance no turno do cliente.',
    instructions=(
        'Analise um único turno em três perspectivas: SPIN, objeções e compliance. '
        'Não invente fatos. secondary_feedback só se houver algo novo vs o feedback principal.'
    ),
    trigger_phases=('neutro', 'situacao', 'problema', 'implicacao', 'necessidade'),
    model='gemini-2.5-flash',
    max_latency_ms=8000,
    priority=10,
)
def spin_coach_prompt(ctx: SpecialistTurnContext, spec: SpecialistDef) -> str:
    return (
        'Analise um único turno de uma conversa de vendas em três perspectivas: '
        'SPIN, objeções e compliance. Use somente a evidência e o estado fornecidos. '
        'Não invente fatos. Retorne JSON puro com exatamente os campos: '
        'source_turn_id, fase_spin, proxima_pergunta_spin, alerta_risco_spin, '
        'objecoes_detectadas, objection_hint, compliance_flagged, '
        'compliance_severity, compliance_reason, evidence_text, '
        'secondary_feedback, secondary_feedback_type, confidence, next_turn_hint. '
        'confidence DEVE ser número entre 0.0 e 1.0 (nunca texto). '
        'compliance_severity DEVE ser info, warning ou critical (use info quando '
        'compliance_flagged=false). compliance_reason DEVE ser string (use "" se vazio). '
        'secondary_feedback deve ser curto e acionável; deixe vazio se não houver '
        'algo novo em relação ao feedback principal. Tipos permitidos: '
        'risk, objection, clarification.\n\n'
        f'source_turn_id={ctx.turn_id}\n'
        f'evidencia={ctx.evidence_text[:1000]}\n'
        f'feedback_principal={ctx.primary_feedback[:500]}\n'
        f'contexto_host={ctx.host_context[:1500]}\n'
        f'estado={json.dumps(ctx.conversation_state, ensure_ascii=False)}'
    )
