"""Tests for utterance completeness gate."""

from __future__ import annotations

from src.modules.transcription.utterance_completeness import (
    CompletenessVerdict,
    evaluate_utterance_completeness,
    text_fingerprint,
)


def test_trailing_connective_blocks_incomplete() -> None:
    result = evaluate_utterance_completeness(
        'Não consigo comprar o produto mas',
        min_words=5,
        now_wall_ms=10_000,
        last_growth_wall_ms=9_000,
        growth_window_ms=400,
    )
    assert result.verdict == CompletenessVerdict.INCOMPLETE
    assert result.reason == 'trailing_connective'


def test_terminal_punctuation_ready() -> None:
    result = evaluate_utterance_completeness(
        'Não consigo comprar o produto certo no momento.',
        min_words=5,
        now_wall_ms=10_000,
        last_growth_wall_ms=9_000,
        growth_window_ms=400,
    )
    assert result.verdict == CompletenessVerdict.READY
    assert result.reason == 'terminal_punctuation'


def test_recent_growth_blocks() -> None:
    result = evaluate_utterance_completeness(
        'Achei muito caro comparado ao concorrente.',
        min_words=5,
        now_wall_ms=10_000,
        last_growth_wall_ms=9_900,
        growth_window_ms=400,
    )
    assert result.verdict == CompletenessVerdict.INCOMPLETE
    assert result.reason == 'recent_growth'


def test_text_fingerprint_normalizes() -> None:
    a = text_fingerprint('Olá, mundo!')
    b = text_fingerprint('ola mundo')
    assert a == b
