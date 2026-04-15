"""Unit tests for LLM state validation and conversation state management."""

import pytest
from src.modules.text_analysis.llm_state_validator import (
    ConversationState,
    LLMAnalysisResult,
    validate_conversation_state,
    validate_llm_response,
    VALID_OBJECTION_CATEGORIES,
)


class TestConversationState:
    """Test conversation state validation."""

    def test_default_state(self):
        """Test default state initialization."""
        state = ConversationState.default_state()
        
        assert state.interesse == "medio"
        assert state.resistencia == "baixa"
        assert state.objecoes_detectadas == []
        assert state.engajamento == "medio"
        assert state.fase_spin == "neutro"
        assert state.proxima_pergunta_spin == ""
        assert state.alerta_risco_spin is False

    def test_valid_objections_are_kept(self):
        """Test that valid objection categories are preserved."""
        state = ConversationState(
            objecoes_detectadas=["preco", "concorrente", "tempo"]
        )
        
        assert state.objecoes_detectadas == ["preco", "concorrente", "tempo"]

    def test_invalid_objections_are_filtered(self):
        """Test that invalid objection categories are filtered out."""
        state = ConversationState(
            objecoes_detectadas=["preco", "invalid_objection", "tempo", "foo"]
        )
        
        assert "preco" in state.objecoes_detectadas
        assert "tempo" in state.objecoes_detectadas
        assert "invalid_objection" not in state.objecoes_detectadas
        assert "foo" not in state.objecoes_detectadas

    def test_case_insensitive_objections(self):
        """Test that objection validation is case-insensitive."""
        state = ConversationState(
            objecoes_detectadas=["PRECO", "Concorrente", "TEMPO"]
        )
        
        assert len(state.objecoes_detectadas) == 3
        assert "preco" in state.objecoes_detectadas
        assert "concorrente" in state.objecoes_detectadas
        assert "tempo" in state.objecoes_detectadas

    def test_invalid_fase_spin_becomes_neutro(self):
        """Unknown fase_spin values are coerced to neutro."""
        state = ConversationState(fase_spin="invalid_phase")
        assert state.fase_spin == "neutro"

    def test_to_dict_serialization(self):
        """Test state to dict conversion."""
        state = ConversationState(
            interesse="alto",
            resistencia="media",
            objecoes_detectadas=["preco"],
            engajamento="alto"
        )
        
        result = state.to_dict()
        
        assert result["interesse"] == "alto"
        assert result["resistencia"] == "media"
        assert result["objecoes_detectadas"] == ["preco"]
        assert result["engajamento"] == "alto"
        assert result["fase_spin"] == "neutro"
        assert result["proxima_pergunta_spin"] == ""
        assert result["alerta_risco_spin"] is False

    def test_validate_conversation_state_with_valid_dict(self):
        """Test validation with valid dict input."""
        raw_state = {
            "interesse": "alto",
            "resistencia": "alta",
            "objecoes_detectadas": ["preco", "concorrente"],
            "engajamento": "medio",
            "fase_spin": "problema",
            "proxima_pergunta_spin": "Quanto isso custa hoje para vocês?",
            "alerta_risco_spin": False,
        }
        
        state = validate_conversation_state(raw_state)
        
        assert state.interesse == "alto"
        assert state.resistencia == "alta"
        assert len(state.objecoes_detectadas) == 2
        assert state.fase_spin == "problema"
        assert "custa" in state.proxima_pergunta_spin

    def test_validate_conversation_state_with_invalid_dict(self):
        """Test validation with invalid dict falls back to defaults."""
        raw_state = {
            "interesse": "invalid_value",
            "resistencia": "also_invalid",
            "objecoes_detectadas": "not_a_list",
            "engajamento": 123
        }
        
        state = validate_conversation_state(raw_state)
        
        # Should fall back to defaults
        assert state.interesse == "medio"
        assert state.resistencia == "baixa"


class TestLLMAnalysisResult:
    """Test LLM analysis result validation."""

    def test_default_values(self):
        """Test default result initialization."""
        result = LLMAnalysisResult()
        
        assert result.feedback is None
        assert result.confidence == 0.5
        assert result.feedback_type is None
        assert isinstance(result.estado, ConversationState)

    def test_direct_feedback_property(self):
        """Test direct_feedback property."""
        result = LLMAnalysisResult(feedback="Test feedback")
        assert result.direct_feedback == "Test feedback"
        
        result = LLMAnalysisResult(feedback=None)
        assert result.direct_feedback == ""
        
        result = LLMAnalysisResult(feedback="  ")
        assert result.direct_feedback == ""

    def test_confidence_bounds(self):
        """Test confidence must stay within 0.0-1.0 (Pydantic field constraints)."""
        assert LLMAnalysisResult(confidence=1.0).confidence == 1.0
        assert LLMAnalysisResult(confidence=0.0).confidence == 0.0

    def test_feedback_type_validation(self):
        """Test feedback_type validation."""
        result = LLMAnalysisResult(feedback_type="objection")
        assert result.feedback_type == "objection"
        
        result = LLMAnalysisResult(feedback_type="invalid_type")
        assert result.feedback_type is None
        
        result = LLMAnalysisResult(feedback_type=None)
        assert result.feedback_type is None

    def test_conversation_state_json(self):
        """Test conversation state JSON serialization."""
        result = LLMAnalysisResult(
            estado=ConversationState(
                interesse="alto",
                objecoes_detectadas=["preco"]
            )
        )
        
        json_str = result.conversation_state_json
        assert "alto" in json_str
        assert "preco" in json_str

    def test_validate_llm_response_with_complete_dict(self):
        """Test validation with complete response dict."""
        raw_response = {
            "feedback": "Test feedback",
            "confidence": 0.85,
            "feedback_type": "objection",
            "estado": {
                "interesse": "medio",
                "resistencia": "alta",
                "objecoes_detectadas": ["preco"],
                "engajamento": "medio"
            }
        }
        
        result = validate_llm_response(raw_response)
        
        assert result.feedback == "Test feedback"
        assert result.confidence == 0.85
        assert result.feedback_type == "objection"
        assert result.estado.resistencia == "alta"

    def test_validate_llm_response_with_missing_fields(self):
        """Test validation with missing fields."""
        raw_response = {"feedback": "Test"}
        
        result = validate_llm_response(raw_response)
        
        assert result.feedback == "Test"
        assert result.confidence == 0.5  # default
        assert result.feedback_type is None
        assert isinstance(result.estado, ConversationState)

    def test_validate_llm_response_with_invalid_estado(self):
        """Test validation with invalid estado."""
        raw_response = {
            "feedback": "Test",
            "estado": "invalid"
        }
        
        result = validate_llm_response(raw_response)
        
        assert result.feedback == "Test"
        assert isinstance(result.estado, ConversationState)
