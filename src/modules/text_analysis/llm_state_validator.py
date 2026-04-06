"""Pydantic models for LLM conversation state validation and response schema."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


# Valid values for state properties
INTEREST_LEVEL = Literal["baixo", "medio", "alto"]
RESISTANCE_LEVEL = Literal["baixa", "media", "alta"]
ENGAGEMENT_LEVEL = Literal["baixo", "medio", "alto"]

# Predefined objection categories (prevents LLM from inventing random categories)
VALID_OBJECTION_CATEGORIES = frozenset({
    "preco",           # Price/cost concerns
    "concorrente",     # Competitor comparisons
    "tempo",           # Timing objections
    "confianca",       # Trust/credibility issues
    "funcionalidade",  # Feature limitations
    "contrato",        # Contract terms
    "implementacao",   # Implementation concerns
    "roi",            # ROI doubts
})


class ConversationState(BaseModel):
    """Validated conversation state model.
    
    This model ensures the LLM returns properly structured state
    and prevents corruption from malformed responses.
    """
    
    interesse: INTEREST_LEVEL = Field(
        default="medio",
        description="Client interest level: baixo, medio, or alto"
    )
    resistencia: RESISTANCE_LEVEL = Field(
        default="baixa",
        description="Client resistance level: baixa, media, or alta"
    )
    objecoes_detectadas: list[str] = Field(
        default_factory=list,
        description="List of detected objection categories"
    )
    engajamento: ENGAGEMENT_LEVEL = Field(
        default="medio",
        description="Client engagement level: baixo, medio, or alto"
    )
    
    @field_validator("objecoes_detectadas")
    @classmethod
    def validate_objections(cls, v: list[str]) -> list[str]:
        """Filter objections to only include valid categories.
        
        This prevents the LLM from inventing random objection types
        and ensures consistency across all feedback events.
        """
        return [
            obj.lower().strip()
            for obj in v
            if obj.lower().strip() in VALID_OBJECTION_CATEGORIES
        ]
    
    @field_validator("interesse", "engajamento")
    @classmethod
    def validate_level_fields(cls, v: str) -> str:
        """Validate level fields are properly normalized."""
        v = v.lower().strip()
        if v not in ("baixo", "medio", "alto"):
            raise ValueError(f"Invalid level value: {v}. Must be 'baixo', 'medio', or 'alto'")
        return v
    
    @field_validator("resistencia")
    @classmethod
    def validate_resistance(cls, v: str) -> str:
        """Validate resistance field is properly normalized."""
        v = v.lower().strip()
        if v not in ("baixa", "media", "alta"):
            raise ValueError(f"Invalid resistance value: {v}. Must be 'baixa', 'media', or 'alta'")
        return v
    
    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return {
            "interesse": self.interesse,
            "resistencia": self.resistencia,
            "objecoes_detectadas": self.objecoes_detectadas,
            "engajamento": self.engajamento,
        }
    
    @classmethod
    def default_state(cls) -> ConversationState:
        """Create a default initial state for a new conversation."""
        return cls()


class LLMAnalysisResult(BaseModel):
    """Validated LLM analysis result.
    
    Ensures the Gemini response has the correct structure before
    processing it downstream.
    """
    
    feedback: str | None = Field(
        default=None,
        description="Tactical feedback suggestion for the seller (1-2 sentences)"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM confidence in this analysis (0.0 to 1.0)"
    )
    feedback_type: str | None = Field(
        default=None,
        description="Type of feedback: objection, opportunity, rapport, closing, or null"
    )
    estado: ConversationState = Field(
        default_factory=ConversationState.default_state,
        description="Updated conversation state"
    )
    
    @field_validator("feedback_type")
    @classmethod
    def validate_feedback_type(cls, v: str | None) -> str | None:
        """Validate feedback type is one of the allowed values."""
        if v is None:
            return None
        v = v.lower().strip()
        valid_types = {"objection", "opportunity", "rapport", "closing", "clarification", "risk"}
        if v not in valid_types:
            return None  # Silently ignore invalid types
        return v
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range."""
        return max(0.0, min(1.0, v))
    
    @property
    def direct_feedback(self) -> str:
        """Get feedback as string (empty if None)."""
        return self.feedback.strip() if self.feedback else ""
    
    @property
    def conversation_state_json(self) -> str:
        """Get conversation state as JSON string."""
        import json
        return json.dumps(self.estado.to_dict(), ensure_ascii=False)


def validate_conversation_state(raw_state: dict) -> ConversationState:
    """Validate and normalize a raw conversation state dict from LLM.
    
    Returns a valid ConversationState, falling back to defaults if invalid.
    """
    try:
        return ConversationState(**raw_state)
    except Exception:
        # Return safe defaults instead of crashing
        return ConversationState.default_state()


def validate_llm_response(raw_response: dict) -> LLMAnalysisResult:
    """Validate and normalize a raw LLM response dict.
    
    Returns a valid LLMAnalysisResult, falling back to safe defaults if invalid.
    """
    try:
        # Extract estado if present, otherwise use default
        estado_raw = raw_response.get("estado", {})
        estado = validate_conversation_state(estado_raw)
        
        return LLMAnalysisResult(
            feedback=raw_response.get("feedback"),
            confidence=raw_response.get("confidence", 0.5),
            feedback_type=raw_response.get("feedback_type"),
            estado=estado,
        )
    except Exception:
        # Return safe fallback
        return LLMAnalysisResult(
            feedback="",
            confidence=0.0,
            feedback_type=None,
            estado=ConversationState.default_state(),
        )
