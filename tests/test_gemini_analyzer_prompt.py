"""Regression tests for Gemini prompt construction (f-string escaping)."""

from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer
from src.modules.text_analysis.llm_state_validator import ConversationState


def test_build_prompt_no_fstring_value_error():
    """Literal JSON braces in the prompt must not break f-string compilation."""
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    prompt = analyzer._build_prompt(
        "O concorrente X está mais barato.",
        ConversationState.default_state().to_dict(),
    )
    assert "playbook_template_key" in prompt
    assert '"estado": {}' in prompt
    assert '"feedback": null' in prompt
    assert "O concorrente X está mais barato." in prompt


def test_build_prompt_marks_host_context_only():
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    prompt = analyzer._build_prompt(
        "Nossa solução reduz o tempo de implantação.",
        ConversationState.default_state().to_dict(),
        speaker_role="host",
    )

    assert "PAPEL DO TRECHO: vendedor/host" in prompt
    assert "NÃO gere feedback" in prompt
