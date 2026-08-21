"""Prompt wrapper shared with the Nest dry-run compiler."""

from __future__ import annotations

from typing import Any

from .types import SpecialistDef, SpecialistTurnContext


def compile_specialist_prompt(
    spec: SpecialistDef,
    ctx: SpecialistTurnContext,
) -> str:
    if spec.prompt_fn is not None:
        return spec.prompt_fn(
            {
                'spec': spec,
                'ctx': ctx,
            },
        )
    tone = (spec.tone or 'direto, curto, acionável').strip()
    example = (spec.example_message or '').strip()
    parts = [
        f'Você é o especialista "{spec.name}" em uma conversa de vendas ao vivo.',
        f'O que observa: {spec.description or spec.instructions}',
        f'Instruções: {spec.instructions}' if spec.instructions else '',
        f'Tom: {tone}.',
        f'Exemplo de mensagem boa: {example}' if example else '',
        'Use SOMENTE a evidência. Não invente fatos. Não fale com o cliente.',
        'Retorne JSON puro com exatamente:',
        'source_turn_id (string), secondary_feedback (string curta ou ""),',
        'secondary_feedback_type (risk|objection|clarification),',
        'confidence (número 0..1), evidence_text (citação literal curta),',
        'next_turn_hint (string, dica para o próximo turno ou "").',
        '',
        f'source_turn_id={ctx.turn_id}',
        f'evidencia={ctx.evidence_text[:2000]}',
        f'feedback_principal={ctx.primary_feedback[:500]}',
        f'contexto_host={ctx.host_context[:1500]}',
        f'estado={_dumps(ctx.conversation_state)}',
    ]
    return '\n'.join(p for p in parts if p)


def _dumps(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return '{}'
