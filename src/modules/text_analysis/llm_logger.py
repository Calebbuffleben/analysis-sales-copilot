"""Structured logging for LLM interactions with correlation IDs.

Provides consistent log formatting that can be grepped across Python and NestJS
to trace a single feedback event from audio capture to UI display.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger('llm.analysis')


def log_llm_interaction(
    meeting_id: str,
    participant_id: str,
    window_end_ms: int,
    transcript: str,
    analysis_result: Dict[str, Any],
    duration_ms: float,
    cache_hit: bool = False,
    fallback_used: bool = False,
    error: Optional[str] = None,
) -> None:
    """Log a complete LLM analysis interaction.
    
    This creates a single structured log line that can be correlated with
    Python pipeline logs and NestJS backend logs via the trace_id.
    
    Args:
        meeting_id: Meeting identifier
        participant_id: Participant identifier  
        window_end_ms: End timestamp of the audio window
        transcript: Transcribed text that was analyzed
        analysis_result: The analysis result dict
        duration_ms: How long the analysis took
        cache_hit: Whether this was a cache hit
        fallback_used: Whether rule-based fallback was used
        error: Error message if analysis failed
    """
    # Generate deterministic trace_id (same algorithm as backend)
    trace_id = make_llm_trace_id(meeting_id, participant_id, window_end_ms)
    
    log_entry = {
        'stage': 'llm.analysis',
        'trace_id': trace_id,
        'meeting_id': meeting_id,
        'participant_id': participant_id,
        'window_end_ms': window_end_ms,
        'duration_ms': round(duration_ms, 2),
        'cache_hit': cache_hit,
        'fallback_used': fallback_used,
        'input_chars': len(transcript),
        'input_preview': transcript[:100] if transcript else '',
    }
    
    if error:
        log_entry['error'] = error
        log_entry['outcome'] = 'error'
    else:
        log_entry['outcome'] = 'success' if analysis_result.get('direct_feedback') else 'no_feedback'
        log_entry['output_feedback'] = analysis_result.get('direct_feedback', '')
        log_entry['output_confidence'] = analysis_result.get('confidence', 0.0)
        log_entry['output_feedback_type'] = analysis_result.get('feedback_type')
    
    logger.info(json.dumps(log_entry, ensure_ascii=False))


def log_llm_state_change(
    meeting_id: str,
    participant_id: str,
    old_state: Dict[str, Any],
    new_state: Dict[str, Any],
) -> None:
    """Log when conversation state changes significantly.
    
    Only logs when there are meaningful changes (not every micro-adjustment).
    """
    # Detect significant changes
    significant_changes = []
    
    if old_state.get('interesse') != new_state.get('interesse'):
        significant_changes.append(
            f"interesse: {old_state.get('interesse')} -> {new_state.get('interesse')}"
        )
    
    if old_state.get('resistencia') != new_state.get('resistencia'):
        significant_changes.append(
            f"resistencia: {old_state.get('resistencia')} -> {new_state.get('resistencia')}"
        )
    
    if old_state.get('engajamento') != new_state.get('engajamento'):
        significant_changes.append(
            f"engajamento: {old_state.get('engajamento')} -> {new_state.get('engajamento')}"
        )
    
    old_objections = set(old_state.get('objecoes_detectadas', []))
    new_objections = set(new_state.get('objecoes_detectadas', []))
    if old_objections != new_objections:
        added = new_objections - old_objections
        removed = old_objections - new_objections
        if added:
            significant_changes.append(f"new objections: {', '.join(added)}")
        if removed:
            significant_changes.append(f"resolved objections: {', '.join(removed)}")
    
    # Only log if there are significant changes
    if significant_changes:
        trace_id = make_llm_trace_id(meeting_id, participant_id, int(time.time() * 1000))
        
        logger.info(
            json.dumps({
                'stage': 'llm.state_change',
                'trace_id': trace_id,
                'meeting_id': meeting_id,
                'participant_id': participant_id,
                'changes': significant_changes,
                'new_state': new_state,
            }, ensure_ascii=False)
        )


def make_llm_trace_id(meeting_id: str, participant_id: str, window_end_ms: int) -> str:
    """Generate a deterministic trace ID for correlation across services.
    
    Uses the same algorithm as backend/src/feedback/feedback-trace.ts
    and python-service/src/feedback_trace.py to ensure grep-ability.
    
    Returns: First 12 hex chars of SHA-256(meeting_id|participant_id|window_end_ms)
    """
    import hashlib
    payload = f"{meeting_id}|{participant_id}|{window_end_ms}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
