"""Provider-neutral contracts for the realtime coaching Live path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionContext:
    meeting_id: str
    tenant_id: str
    sample_rate: int = 16_000
    channels: int = 1
    catalog_prompt: str = ''
    resumption_handle: Optional[str] = None
    context_window_tokens: int = 12_000
    api_key: str = ''


@dataclass
class CoachTurnResult:
    """Normalized emit_feedback payload — independent of the Live vendor."""

    turn_id: str
    feedback: str
    confidence: float
    feedback_type: str
    evidence_text: str
    conversation_state: dict[str, Any] = field(default_factory=dict)
    playbook_template_key: str = ''
    playbook_variables: dict[str, Any] = field(default_factory=dict)
    raw_args: dict[str, Any] = field(default_factory=dict)


def turn_result_from_tool_args(args: dict[str, Any]) -> CoachTurnResult:
    estado = args.get('estado') if isinstance(args.get('estado'), dict) else {}
    variables = args.get('playbook_variables')
    return CoachTurnResult(
        turn_id=str(args.get('turnId') or args.get('turn_id') or ''),
        feedback=str(args.get('feedback') or ''),
        confidence=float(args.get('confidence') or 0.0),
        feedback_type=str(args.get('feedback_type') or ''),
        evidence_text=str(args.get('evidence_text') or ''),
        conversation_state=dict(estado or {}),
        playbook_template_key=str(args.get('playbook_template_key') or ''),
        playbook_variables=dict(variables) if isinstance(variables, dict) else {},
        raw_args=dict(args),
    )
