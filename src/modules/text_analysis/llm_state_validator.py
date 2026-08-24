"""Pydantic models for LLM conversation state validation and response schema."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


# Valid values for state properties
INTEREST_LEVEL = Literal["baixo", "medio", "alto"]
RESISTANCE_LEVEL = Literal["baixa", "media", "alta"]
ENGAGEMENT_LEVEL = Literal["baixo", "medio", "alto"]

SENTIMENTO = Literal["positivo", "neutro", "negativo"]
SENTIMENTO_TENDENCIA = Literal["subindo", "estavel", "caindo"]
FASE_SPIN = Literal["neutro", "situacao", "problema", "implicacao", "necessidade"]

_VALID_FASE_SPIN = frozenset({"neutro", "situacao", "problema", "implicacao", "necessidade"})

_FASE_SPIN_ALIASES = {
    'fechamento': 'necessidade',
    'closing': 'necessidade',
    'close': 'necessidade',
    'situação': 'situacao',
    'situation': 'situacao',
    'problem': 'problema',
    'implication': 'implicacao',
    'implicacao': 'implicacao',
    'need': 'necessidade',
    'need-payoff': 'necessidade',
}


def normalize_fase_spin(value: object) -> str:
    """Coerce LLM SPIN phase labels to the canonical set."""
    if value is None:
        return 'neutro'
    s = str(value).lower().strip()
    if s in _VALID_FASE_SPIN:
        return s
    aliased = _FASE_SPIN_ALIASES.get(s)
    if aliased:
        return aliased
    return 'neutro'


# Max length for suggested SPIN question (prompt / UI safety)
_PROXIMA_PERGUNTA_SPIN_MAX_LEN = 500
_MEETING_PRODUCT_MAX_LEN = 200
_MEETING_CONTEXT_ITEM_MAX_LEN = 120
_MEETING_CONTEXT_MAX_ITEMS = 20

# Aligned with backend PLAYBOOK_MAX_TEMPLATE_KEY_CHARS / payload caps
_PLAYBOOK_MAX_TEMPLATE_KEY_CHARS = 64
_PLAYBOOK_MAX_VARIABLE_KEYS = 32
_PLAYBOOK_MAX_VAR_KEY_CHARS = 64
_PLAYBOOK_MAX_VAR_VALUE_CHARS = 2000

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


def normalize_playbook_template_key(raw: object) -> str | None:
    """Normalize LLM hint template key; None if empty after trim."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if len(s) > _PLAYBOOK_MAX_TEMPLATE_KEY_CHARS:
        return s[:_PLAYBOOK_MAX_TEMPLATE_KEY_CHARS]
    return s


def normalize_playbook_variables(raw: object) -> dict[str, str]:
    """Coerce playbook_variables to a bounded dict[str, str]."""
    if not raw or not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if len(out) >= _PLAYBOOK_MAX_VARIABLE_KEYS:
            break
        ks = str(k).strip()
        if len(ks) > _PLAYBOOK_MAX_VAR_KEY_CHARS:
            ks = ks[:_PLAYBOOK_MAX_VAR_KEY_CHARS]
        if not ks:
            continue
        if v is None:
            vs = ""
        else:
            vs = str(v).strip()
        if len(vs) > _PLAYBOOK_MAX_VAR_VALUE_CHARS:
            vs = vs[:_PLAYBOOK_MAX_VAR_VALUE_CHARS]
        out[ks] = vs
    return out


def build_playbook_hint_json(
    template_key: str | None,
    variables: dict[str, str] | None,
) -> str:
    """Serialize JSON for AnalysisPayload.playbook_hint_json; empty if no key."""
    key = normalize_playbook_template_key(template_key)
    if not key:
        return ""
    import json

    vars_norm = normalize_playbook_variables(variables or {})
    return json.dumps(
        {
            "playbook_template_key": key,
            "playbook_variables": vars_norm,
        },
        ensure_ascii=False,
    )


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
    fase_spin: FASE_SPIN = Field(
        default="neutro",
        description="SPIN phase: neutro until clearly detected; then situacao/problema/implicacao/necessidade",
    )
    proxima_pergunta_spin: str = Field(
        default="",
        description="Suggested next SPIN question when phase is known; empty when neutro or not applicable",
    )
    alerta_risco_spin: bool = Field(
        default=False,
        description="True when seller skips SPIN steps (e.g. solution before problem/implication)",
    )
    product: str = Field(
        default="",
        description="Product or solution being discussed in the meeting",
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Explicit customer pain points accumulated across the meeting",
    )
    objections: list[str] = Field(
        default_factory=list,
        description="Free-form objections accumulated across the meeting",
    )
    claims: list[str] = Field(
        default_factory=list,
        description="Benefits, claims, or promises stated during the meeting",
    )
    sentimento_cliente: SENTIMENTO = Field(
        default="neutro",
        description="Customer sentiment: positivo, neutro, or negativo",
    )
    sentimento_tendencia: SENTIMENTO_TENDENCIA = Field(
        default="estavel",
        description="Sentiment trend: subindo, estavel, or caindo",
    )
    objecoes_ativas: list[str] = Field(
        default_factory=list,
        description="Active unresolved objection categories",
    )
    objecoes_resolvidas: list[str] = Field(
        default_factory=list,
        description="Resolved objection categories",
    )

    @field_validator("fase_spin", mode="before")
    @classmethod
    def normalize_fase_spin_field(cls, v) -> str:
        return normalize_fase_spin(v)

    @field_validator("proxima_pergunta_spin")
    @classmethod
    def truncate_proxima_pergunta(cls, v: str) -> str:
        s = (v or "").strip()
        if len(s) > _PROXIMA_PERGUNTA_SPIN_MAX_LEN:
            return s[:_PROXIMA_PERGUNTA_SPIN_MAX_LEN]
        return s

    @field_validator("alerta_risco_spin", mode="before")
    @classmethod
    def coerce_alerta_risco(cls, v) -> bool:
        if v is True or v == 1 or str(v).lower() in ("true", "1", "yes"):
            return True
        return False

    @field_validator("product")
    @classmethod
    def truncate_product(cls, v: str) -> str:
        s = (v or "").strip()
        if len(s) > _MEETING_PRODUCT_MAX_LEN:
            return s[:_MEETING_PRODUCT_MAX_LEN]
        return s

    @field_validator("pain_points", "objections", "claims", mode="before")
    @classmethod
    def normalize_context_list(cls, v) -> list[str]:
        """Coerce meeting-context lists to short, bounded, deduped strings."""
        if not v:
            return []
        items = v if isinstance(v, list) else [v]
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            s = str(item).strip()
            if not s:
                continue
            if len(s) > _MEETING_CONTEXT_ITEM_MAX_LEN:
                s = s[:_MEETING_CONTEXT_ITEM_MAX_LEN]
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= _MEETING_CONTEXT_MAX_ITEMS:
                break
        return out

    @field_validator("objecoes_detectadas", "objecoes_ativas", "objecoes_resolvidas")
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
    
    @field_validator("sentimento_cliente", mode="before")
    @classmethod
    def normalize_sentimento(cls, v) -> str:
        s = str(v or "neutro").lower().strip()
        if s in ("positivo", "neutro", "negativo"):
            return s
        aliases = {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}
        return aliases.get(s, "neutro")

    @field_validator("sentimento_tendencia", mode="before")
    @classmethod
    def normalize_tendencia(cls, v) -> str:
        s = str(v or "estavel").lower().strip()
        if s in ("subindo", "estavel", "caindo"):
            return s
        aliases = {
            "up": "subindo",
            "rising": "subindo",
            "down": "caindo",
            "falling": "caindo",
            "stable": "estavel",
            "estável": "estavel",
        }
        return aliases.get(s, "estavel")

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return {
            "interesse": self.interesse,
            "resistencia": self.resistencia,
            "objecoes_detectadas": self.objecoes_detectadas,
            "engajamento": self.engajamento,
            "fase_spin": self.fase_spin,
            "proxima_pergunta_spin": self.proxima_pergunta_spin,
            "alerta_risco_spin": self.alerta_risco_spin,
            "product": self.product,
            "pain_points": self.pain_points,
            "objections": self.objections,
            "claims": self.claims,
            "sentimento_cliente": self.sentimento_cliente,
            "sentimento_tendencia": self.sentimento_tendencia,
            "objecoes_ativas": self.objecoes_ativas,
            "objecoes_resolvidas": self.objecoes_resolvidas,
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
    evidence_text: str = Field(
        default="",
        max_length=1000,
        description="Short literal evidence extracted from audio, empty when unavailable",
    )
    estado: ConversationState = Field(
        default_factory=ConversationState.default_state,
        description="Updated conversation state"
    )
    playbook_template_key: str | None = Field(
        default=None,
        description="Tenant playbook template slug for resolver (optional)",
    )
    playbook_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variables for {{placeholder}} interpolation in template steps",
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

    @field_validator("playbook_template_key", mode="before")
    @classmethod
    def validate_playbook_template_key(cls, v) -> str | None:
        return normalize_playbook_template_key(v)

    @field_validator("playbook_variables", mode="before")
    @classmethod
    def validate_playbook_variables(cls, v) -> dict[str, str]:
        return normalize_playbook_variables(v)
    
    @property
    def direct_feedback(self) -> str:
        """Get feedback as string (empty if None)."""
        return self.feedback.strip() if self.feedback else ""
    
    @property
    def conversation_state_json(self) -> str:
        """Get conversation state as JSON string."""
        import json
        return json.dumps(self.estado.to_dict(), ensure_ascii=False)

    @property
    def playbook_hint_json(self) -> str:
        """JSON string for gRPC playbook_hint_json (empty when no template key)."""
        return build_playbook_hint_json(
            self.playbook_template_key,
            self.playbook_variables,
        )


def _playbook_fields_from_raw(raw_response: dict) -> tuple[str | None, dict[str, str]]:
    """Extract playbook hint from raw LLM JSON (supports alternate key names)."""
    key_raw = (
        raw_response.get("playbook_template_key")
        or raw_response.get("template_key")
        or raw_response.get("playbookTemplateKey")
    )
    vars_raw = (
        raw_response.get("playbook_variables")
        or raw_response.get("playbookVariables")
        or raw_response.get("variables")
    )
    key = normalize_playbook_template_key(key_raw)
    variables = normalize_playbook_variables(vars_raw)
    return key, variables


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
        p_key, p_vars = _playbook_fields_from_raw(raw_response)

        return LLMAnalysisResult(
            feedback=raw_response.get("feedback"),
            confidence=raw_response.get("confidence", 0.5),
            feedback_type=raw_response.get("feedback_type"),
            evidence_text=raw_response.get("evidence_text") or "",
            estado=estado,
            playbook_template_key=p_key,
            playbook_variables=p_vars,
        )
    except Exception:
        # Return safe fallback
        return LLMAnalysisResult(
            feedback="",
            confidence=0.0,
            feedback_type=None,
            evidence_text="",
            estado=ConversationState.default_state(),
            playbook_template_key=None,
            playbook_variables={},
        )
