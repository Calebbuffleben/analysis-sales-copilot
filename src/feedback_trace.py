"""Deterministic feedback trace id + structured JSON logs (Python service)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping


def make_feedback_trace_id(
    meeting_id: str,
    participant_id: str,
    window_end_ms: int,
) -> str:
    """Short deterministic id shared with Nest (same string + SHA-256 slice)."""
    raw = f'{meeting_id}|{participant_id}|{int(window_end_ms)}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]


def log_feedback_trace(
    logger: logging.Logger,
    level: int,
    stage: str,
    *,
    trace_id: str,
    meeting_id: str,
    participant_id: str,
    window_end_ms: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """One JSON line per call; keep keys stable for grep."""
    payload: dict[str, Any] = {
        'stage': stage,
        'traceId': trace_id,
        'meetingId': meeting_id,
        'participantId': participant_id,
        'windowEndMs': int(window_end_ms),
    }
    if extra:
        for k, v in extra.items():
            if v is not None:
                payload[k] = v
    logger.log(level, json.dumps(payload, default=str, separators=(',', ':')))
