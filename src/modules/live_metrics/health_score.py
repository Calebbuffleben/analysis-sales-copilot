"""Deterministic 0–100 health score from ConversationState + talk stats + prosody."""

from __future__ import annotations

from typing import Any

_LEVEL = {'baixo': 0.0, 'baixa': 0.0, 'medio': 0.5, 'media': 0.5, 'alto': 1.0, 'alta': 1.0}
_SPIN_INDEX = {
    'neutro': 0,
    'situacao': 1,
    'problema': 2,
    'implicacao': 3,
    'necessidade': 4,
}


def _level(value: object) -> float:
    return _LEVEL.get(str(value or '').strip().lower(), 0.5)


def compute_health_score(
    *,
    estado: dict[str, Any] | None,
    host_ratio: float = 0.0,
    energy_level: str = '',
    hesitation_hint: bool = False,
) -> tuple[int, list[str]]:
    """Return (score 0-100, contributing factors). Fail-open to 50 on empty state."""
    state = estado or {}
    score = 50.0
    factors: list[str] = []

    interesse = _level(state.get('interesse'))
    engajamento = _level(state.get('engajamento'))
    resistencia = _level(state.get('resistencia'))
    score += (interesse - 0.5) * 20
    score += (engajamento - 0.5) * 20
    score -= resistencia * 15
    if interesse >= 1.0:
        factors.append('interesse alto')
    if engajamento <= 0.0:
        factors.append('engajamento baixo')
        score -= 5
    if resistencia >= 1.0:
        factors.append('resistência alta')

    active = _active_objections(state)
    if active:
        penalty = min(24, 8 * len(active))
        score -= penalty
        factors.append(f'{len(active)} objeção(ões) ativas')

    if state.get('alerta_risco_spin') is True:
        score -= 15
        factors.append('risco SPIN')

    if host_ratio >= 0.70:
        score -= 10
        factors.append('talk-to-listen alto')

    sentiment = str(state.get('sentimento_cliente') or 'neutro').lower()
    trend = str(state.get('sentimento_tendencia') or 'estavel').lower()
    if sentiment == 'negativo':
        score -= 10
        factors.append('sentimento negativo')
    elif sentiment == 'positivo':
        score += 8
    if trend == 'caindo':
        score -= 12
        factors.append('sentimento caindo')

    if hesitation_hint or energy_level == 'low':
        score -= 5
        factors.append('prosódia hesitante')

    clamped = int(max(0, min(100, round(score))))
    return clamped, factors


def playbook_adherence(
    estado: dict[str, Any] | None,
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = estado or {}
    fase = str(state.get('fase_spin') or 'neutro').lower()
    spin_idx = _SPIN_INDEX.get(fase, 0)
    spin_percent = int(round((spin_idx / 4) * 100)) if spin_idx else 0
    catalog_steps = steps or []
    mapped: list[dict[str, Any]] = []
    if catalog_steps:
        n = max(len(catalog_steps), 1)
        done_upto = min(spin_idx, n)
        for i, step in enumerate(catalog_steps):
            label = str(step.get('label') or step.get('title') or f'Etapa {i + 1}')
            mapped.append({
                'id': str(step.get('id') or i),
                'label': label,
                'done': i < done_upto,
            })
        percent = int(round((done_upto / n) * 100))
    else:
        mapped = [
            {'id': 'situacao', 'label': 'Situação', 'done': spin_idx >= 1},
            {'id': 'problema', 'label': 'Problema', 'done': spin_idx >= 2},
            {'id': 'implicacao', 'label': 'Implicação', 'done': spin_idx >= 3},
            {'id': 'necessidade', 'label': 'Necessidade', 'done': spin_idx >= 4},
        ]
        percent = spin_percent
    return {'percent': percent, 'faseSpin': fase, 'steps': mapped}


def _active_objections(state: dict[str, Any]) -> list[str]:
    active = state.get('objecoes_ativas')
    if isinstance(active, list) and active:
        return [str(x) for x in active]
    detected = state.get('objecoes_detectadas') or []
    resolved = {str(x).lower() for x in (state.get('objecoes_resolvidas') or [])}
    return [str(x) for x in detected if str(x).lower() not in resolved]
