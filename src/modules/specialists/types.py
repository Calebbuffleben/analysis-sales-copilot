"""Specialist catalog types used by registry, LangGraph and the Live publisher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

SpecialistSource = Literal['code', 'custom']


@dataclass
class SpecialistDef:
    key: str
    name: str
    description: str = ''
    instructions: str = ''
    tone: str = ''
    example_message: str = ''
    trigger_phases: tuple[str, ...] = ()
    trigger_keywords: tuple[str, ...] = ()
    min_confidence: float = 0.6
    cooldown_sec: int = 15
    priority: int = 100
    model: str = 'gemini-2.5-flash'
    max_latency_ms: int = 4000
    source: SpecialistSource = 'code'
    enabled: bool = True
    icon: str = ''
    color: str = ''
    prompt_fn: Optional[Callable[[dict[str, Any]], str]] = None


@dataclass
class SpecialistTurnContext:
    tenant_id: str
    meeting_id: str
    participant_id: str
    participant_role: str
    turn_id: str
    speech_end_ms: int
    evidence_text: str
    primary_feedback: str
    conversation_state: dict[str, Any]
    host_context: str
    selected_keys: tuple[str, ...] = ()


@dataclass
class SpecialistOutput:
    key: str
    name: str
    secondary_feedback: str = ''
    secondary_feedback_type: str = 'clarification'
    confidence: float = 0.0
    evidence_text: str = ''
    next_turn_hint: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
