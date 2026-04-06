"""Detection logic for indecision signals in transcribed text."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


class IndecisionSignalDetector:
    """Detect hedging and indecision patterns in text."""

    CONDITIONAL_KEYWORDS = (
        'se',
        'caso',
        'dependendo',
        'desde que',
        'contanto que',
    )

    POSTPONEMENT_KEYWORDS = (
        'depois',
        'mais tarde',
        'semana que vem',
        'mes que vem',
        'mês que vem',
        'amanha',
        'amanhã',
        'vou pensar',
        'preciso pensar',
        'te aviso',
        'te retorno',
        'retorno depois',
        'volto nisso',
        'adiar',
        'adiamento',
        'postergar',
    )

    def analyze(self, text: str) -> Optional[Dict[str, object]]:
        """Return indecision metrics when the text looks indecisive."""
        normalized_text = text.lower()
        matched_conditional = self._find_keywords(
            normalized_text,
            self.CONDITIONAL_KEYWORDS,
        )
        matched_postponement = self._find_keywords(
            normalized_text,
            self.POSTPONEMENT_KEYWORDS,
        )
        question_count = normalized_text.count('?')
        conditional_language_score = min(
            1.0,
            (len(matched_conditional) * 0.5) + (question_count * 0.1),
        )
        postponement_likelihood = min(
            1.0,
            (len(matched_postponement) * 0.6)
            + (0.15 if 'depois' in normalized_text else 0.0),
        )

        if max(conditional_language_score, postponement_likelihood) < 0.45:
            return None

        return {
            'conditional_language_score': round(conditional_language_score, 3),
            'postponement_likelihood': round(postponement_likelihood, 3),
        }

    def detect_conditional_keywords(self, text: str) -> List[str]:
        """Return conditional keywords that appear in the text."""
        return self._find_keywords(text.lower(), self.CONDITIONAL_KEYWORDS)

    def _find_keywords(self, text: str, keywords: tuple[str, ...]) -> List[str]:
        matches: List[str] = []
        for keyword in keywords:
            pattern = re.escape(keyword)
            if re.search(rf'\b{pattern}\b', text):
                matches.append(keyword)
        return matches
