"""Unit tests for rule-based fallback analyzer."""

import pytest
from .rule_based_analyzer import (
    analyze_text_fallback,
    OBJECTION_PATTERNS,
    BUYING_SIGNAL_PATTERNS,
    FallbackAnalysisResult,
)


class TestRuleBasedAnalyzer:
    """Test rule-based text analysis fallback."""

    def test_empty_text_returns_empty_result(self):
        """Test that empty text returns no feedback."""
        result = analyze_text_fallback("", {})
        
        assert result.feedback == ""
        assert result.confidence == 0.0
        assert result.feedback_type is None

    def test_short_text_returns_empty_result(self):
        """Test that very short text (< 10 chars) returns no feedback."""
        result = analyze_text_fallback("ok", {})
        
        assert result.feedback == ""

    def test_price_objection_detection(self):
        """Test price objection pattern detection."""
        text = "Achei caro comparado ao que vocês oferecem"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback != ""
        assert result.feedback_type == "objection"
        assert result.confidence >= 0.7
        assert "preco" in result.state_updates.get("objecoes_detectadas", [])

    def test_time_objection_detection(self):
        """Test time objection pattern detection."""
        text = "Preciso pensar, me liga mês que vem"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback != ""
        assert result.feedback_type == "objection"
        assert result.confidence >= 0.7
        assert "tempo" in result.state_updates.get("objecoes_detectadas", [])

    def test_competitor_mention_detection(self):
        """Test competitor mention pattern detection."""
        text = "O concorrente X oferece um preço melhor"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback != ""
        assert result.feedback_type == "objection"
        assert "concorrente" in result.state_updates.get("objecoes_detectadas", [])

    def test_buying_signal_closing(self):
        """Test closing buying signal detection."""
        text = "Ok, me interessa. Como funciona o próximo passo?"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback != ""
        assert result.feedback_type == "closing"
        assert result.confidence >= 0.8
        assert result.state_updates.get("interesse") == "alto"

    def test_opportunity_signal_detection(self):
        """Test opportunity signal detection."""
        text = "Como isso funciona? Pode me explicar?"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback != ""
        assert result.feedback_type == "opportunity"
        assert result.confidence >= 0.6

    def test_no_signal_returns_empty(self):
        """Test that neutral text returns no feedback."""
        text = "Ok, entendi. Pode continuar explicando"
        result = analyze_text_fallback(text, {})
        
        assert result.feedback == ""
        assert result.confidence == 0.0

    def test_state_updates_on_objection(self):
        """Test that state is properly updated on objection."""
        text = "Achei caro demais"
        current_state = {
            "interesse": "alto",
            "resistencia": "baixa",
            "objecoes_detectadas": [],
            "engajamento": "alto"
        }
        
        result = analyze_text_fallback(text, current_state)
        
        assert result.state_updates["resistencia"] in ["media", "alta"]
        assert "preco" in result.state_updates["objecoes_detectadas"]

    def test_state_updates_on_closing_signal(self):
        """Test that state is properly updated on closing signal."""
        text = "Quero avançar com isso agora"
        current_state = {
            "interesse": "medio",
            "resistencia": "media",
            "objecoes_detectadas": [],
            "engajamento": "medio"
        }
        
        result = analyze_text_fallback(text, current_state)
        
        assert result.state_updates["interesse"] == "alto"
        assert result.state_updates["resistencia"] == "baixa"
        assert result.state_updates["engajamento"] == "alto"

    def test_multiple_patterns_no_false_positives(self):
        """Test that patterns don't create false positives on neutral text."""
        neutral_texts = [
            "Ok, entendi",
            "Pode continuar",
            "Hmm, interessante",
            "Blá blá blá genérico",
            "Sim, com certeza",
            "Não, obrigado",
        ]
        
        for text in neutral_texts:
            result = analyze_text_fallback(text, {})
            # Should not trigger feedback on neutral text
            assert result.feedback == "" or result.confidence < 0.5

    def test_fallback_result_to_dict(self):
        """Test FallbackAnalysisResult serialization."""
        result = FallbackAnalysisResult(
            feedback="Test feedback",
            confidence=0.75,
            feedback_type="objection",
            state_updates={"resistencia": "alta"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["direct_feedback"] == "Test feedback"
        assert result_dict["confidence"] == 0.75
        assert result_dict["feedback_type"] == "objection"
        assert result_dict["conversation_state"]["resistencia"] == "alta"
