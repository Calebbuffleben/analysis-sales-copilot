"""Rule-based text analyzer for fallback when LLM is unavailable.

This module provides deterministic keyword/pattern-based analysis as a fallback
when the Gemini LLM service is down, timing out, or returning errors.

The rules are intentionally conservative - only trigger feedback on clear signals
to avoid false positives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class FallbackAnalysisResult:
    """Result from rule-based fallback analysis."""
    
    feedback: str = ""
    confidence: float = 0.0
    feedback_type: Optional[str] = None
    state_updates: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for downstream consumption."""
        return {
            'direct_feedback': self.feedback,
            'confidence': self.confidence,
            'feedback_type': self.feedback_type,
            'conversation_state': self.state_updates or {}
        }


# Objection patterns - high confidence triggers
OBJECTION_PATTERNS = [
    {
        'category': 'preco',
        'patterns': [
            r'\bcaro\b', r'\bpreço\s*alto\b', r'\bsem\s*valor\b',
            r'\bfora\s*do\s*orçamento\b', r'\binvestimento\s*alto\b',
            r'\bnão\s*compensa\b', r'\bretorno\s*(não|nao)\s*garantido\b',
        ],
        'feedback': 'Objeção de preço detectada - reforce ROI e diferenciais de valor',
        'confidence': 0.75,
    },
    {
        'category': 'concorrente',
        'patterns': [
            r'\bconcorrente\b', r'\bcompetidor\b',
            r'\b[mM]elhor\s*(que|do\s*que)\b.*\boferece[m]?\b',
            r'\boutro\s*fornecedor\b', r'\b[mM]ercado\s*oferece\b',
        ],
        'feedback': 'Cliente mencionou concorrente - diferencie sua solução',
        'confidence': 0.70,
    },
    {
        'category': 'tempo',
        'patterns': [
            r'\bpreciso\s*pensar\b', r'\bme\s*liga\s*(depois|mês|mes)\b',
            r'\bnão\s*é\s*urgente\b', r'\bposso\s*esperar\b',
            r'\bno\s*próximo\s*(mês|mes|trimestre|ano)\b',
        ],
        'feedback': 'Objeção de tempo - crie urgência com benefícios de agir agora',
        'confidence': 0.80,
    },
    {
        'category': 'confianca',
        'patterns': [
            r'\bnão\s*conheço\b', r'\bprimeira\s*vez\b',
            r'\b[nN]unca\s*usei\b', r'\bcomo\s*funciona\b',
            r'\bsegurança\b', r'\bgarantia\b',
        ],
        'feedback': 'Dúvida de confiança - apresente cases e credenciais',
        'confidence': 0.65,
    },
]

# Positive buying signals
BUYING_SIGNAL_PATTERNS = [
    {
        'type': 'closing',
        'patterns': [
            r'\bcomo\s*(que|funciona)\s*o\s*(próximo\s*passo|processo)\b',
            r'\bquero\s*avançar\b', r'\bvamos\s*fechar\b',
            r'\bme\s*interessa\b.*\bpróximo\b', r'\bquando\s*começa[m]?\b',
            r'\bpreciso\s*(do|da)\b.*\bpara\s*(ontem|agora)\b',
        ],
        'feedback': 'Sinal de compra! Apresente próximo passo claro (proposta, contrato)',
        'confidence': 0.85,
    },
    {
        'type': 'opportunity',
        'patterns': [
            r'\bcomo\s*isso\s*funciona\b', r'\bpode\s*me\s*explicar\b',
            r'\bquais\s*são\s*(os|as)\b.*\bbenefícios\b',
            r'\b[vV]ocês\s*(têm|tem)\b.*\bcase\b',
        ],
        'feedback': 'Cliente demonstrando interesse - aprofunde na solução',
        'confidence': 0.60,
    },
]


def analyze_text_fallback(text: str, current_state: Dict[str, Any]) -> FallbackAnalysisResult:
    """Analyze text using rule-based patterns when LLM is unavailable.
    
    This is a conservative fallback - only trigger on clear signals.
    Returns empty feedback if no strong pattern is detected.
    """
    if not text or len(text.strip()) < 10:
        return FallbackAnalysisResult()
    
    text_lower = text.lower()
    
    # Check buying signals first (higher priority)
    for signal in BUYING_SIGNAL_PATTERNS:
        for pattern in signal['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.info(
                    f"Fallback detected {signal['type']} signal: "
                    f"pattern='{pattern[:30]}...', text='{text[:50]}...'"
                )
                return FallbackAnalysisResult(
                    feedback=signal['feedback'],
                    confidence=signal['confidence'],
                    feedback_type=signal['type'],
                    state_updates=_update_state_for_buying_signal(current_state, signal['type'])
                )
    
    # Check objection patterns
    for objection in OBJECTION_PATTERNS:
        for pattern in objection['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.info(
                    f"Fallback detected {objection['category']} objection: "
                    f"pattern='{pattern[:30]}...', text='{text[:50]}...'"
                )
                return FallbackAnalysisResult(
                    feedback=objection['feedback'],
                    confidence=objection['confidence'],
                    feedback_type='objection',
                    state_updates=_update_state_for_objection(
                        current_state,
                        objection['category']
                    )
                )
    
    # No strong signal detected
    return FallbackAnalysisResult()


def _update_state_for_buying_signal(
    current_state: Dict[str, Any],
    signal_type: str
) -> Dict[str, Any]:
    """Update conversation state when a buying signal is detected."""
    state = current_state.copy()
    
    if signal_type == 'closing':
        state['interesse'] = 'alto'
        state['resistencia'] = 'baixa'
        state['engajamento'] = 'alto'
    elif signal_type == 'opportunity':
        if state.get('interesse') == 'baixo':
            state['interesse'] = 'medio'
        state['engajamento'] = 'medio'
    
    return state


def _update_state_for_objection(
    current_state: Dict[str, Any],
    objection_category: str
) -> Dict[str, Any]:
    """Update conversation state when an objection is detected."""
    state = current_state.copy()
    
    # Increase resistance
    if state.get('resistencia') == 'baixa':
        state['resistencia'] = 'media'
    elif state.get('resistencia') == 'media':
        state['resistencia'] = 'alta'
    
    # Add objection to list
    existing_objections = state.get('objecoes_detectadas', [])
    if objection_category not in existing_objections:
        state['objecoes_detectadas'] = existing_objections + [objection_category]
    
    # May decrease engagement for strong objections
    if objection_category in ('preco', 'tempo'):
        if state.get('engajamento') == 'alto':
            state['engajamento'] = 'medio'
    
    return state
