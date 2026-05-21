"""Regression tests for Gemini prompt construction (f-string escaping)."""

from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer
from src.modules.text_analysis.llm_state_validator import ConversationState


def test_build_prompt_no_fstring_value_error():
    """Inline JSON examples must not use single `{` inside the f-string body."""
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    prompt = analyzer._build_prompt(
        "O concorrente X está mais barato.",
        ConversationState.default_state().to_dict(),
    )
    assert "playbook_variables" in prompt
    assert '{"competidor": "Concorrente X", "produto": "Suite Pro"}' in prompt
