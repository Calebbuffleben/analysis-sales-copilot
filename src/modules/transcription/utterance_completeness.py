"""Heuristic utterance completeness for partial STT transcripts (Portuguese-aware)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional

_TRAILING_CONNECTIVES = frozenset(
    {
        'e',
        'mas',
        'porque',
        'entao',
        'então',
        'que',
        'de',
        'da',
        'do',
        'dos',
        'das',
        'um',
        'uma',
        'o',
        'a',
        'os',
        'as',
        'no',
        'na',
        'nos',
        'nas',
        'em',
        'para',
        'por',
        'com',
        'se',
        'ou',
    },
)

_TERMINAL_PUNCTUATION = frozenset({'.', '?', '!'})


class CompletenessVerdict(str, Enum):
    READY = 'ready'
    INCOMPLETE = 'incomplete'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class CompletenessResult:
    verdict: CompletenessVerdict
    reason: str


def normalize_transcript_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip outer punctuation for comparison."""
    collapsed = ' '.join((text or '').split()).strip().lower()
    return collapsed.strip('.,!?;:…')


def tokenize(text: str) -> list[str]:
    normalized = normalize_transcript_text(text)
    if not normalized:
        return []
    return normalized.split()


def last_token(text: str) -> str:
    tokens = tokenize(text)
    return tokens[-1] if tokens else ''


def evaluate_utterance_completeness(
    text: str,
    *,
    min_words: int,
    now_wall_ms: int,
    last_growth_wall_ms: Optional[int],
    growth_window_ms: int,
) -> CompletenessResult:
    """Return whether a stable partial looks complete enough to call the LLM."""
    stripped = (text or '').strip()
    if not stripped:
        return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'empty')

    if (
        last_growth_wall_ms is not None
        and growth_window_ms > 0
        and (now_wall_ms - last_growth_wall_ms) < growth_window_ms
    ):
        return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'recent_growth')

    words = tokenize(stripped)
    if len(words) < min_words:
        return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'min_words')

    trailing = last_token(stripped)
    if trailing in _TRAILING_CONNECTIVES:
        return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'trailing_connective')

    if stripped.endswith(','):
        return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'trailing_comma')

    if stripped[-1] in _TERMINAL_PUNCTUATION:
        return CompletenessResult(CompletenessVerdict.READY, 'terminal_punctuation')

    if len(words) >= max(min_words + 2, 7):
        return CompletenessResult(CompletenessVerdict.READY, 'long_stable_phrase')

    return CompletenessResult(CompletenessVerdict.INCOMPLETE, 'no_terminal_punctuation')


def text_fingerprint(text: str) -> str:
    """Normalized fingerprint for dedup across partial/final."""
    normalized = normalize_transcript_text(text)
    normalized = unicodedata.normalize('NFKD', normalized)
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return ' '.join(normalized.split())
