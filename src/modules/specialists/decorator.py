"""Decorator + in-process registry for code-defined specialists."""

from __future__ import annotations

import logging
import pkgutil
from typing import Any, Callable, Optional

from .types import SpecialistDef

logger = logging.getLogger(__name__)

_BUILTINS: dict[str, SpecialistDef] = {}


def specialist(
    *,
    key: str,
    name: str,
    description: str = '',
    instructions: str = '',
    trigger_phases: tuple[str, ...] = (),
    trigger_keywords: tuple[str, ...] = (),
    model: str = 'gemini-2.5-flash',
    max_latency_ms: int = 4000,
    priority: int = 100,
    min_confidence: float = 0.6,
    cooldown_sec: int = 15,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a code specialist. The wrapped function returns a prompt string."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        _BUILTINS[key] = SpecialistDef(
            key=key,
            name=name,
            description=description,
            instructions=instructions,
            trigger_phases=trigger_phases,
            trigger_keywords=trigger_keywords,
            model=model,
            max_latency_ms=max_latency_ms,
            priority=priority,
            min_confidence=min_confidence,
            cooldown_sec=cooldown_sec,
            source='code',
            prompt_fn=lambda payload: fn(payload['ctx'], payload['spec']),
        )
        return fn

    return decorate


def load_builtins() -> dict[str, SpecialistDef]:
    """Import python-service/src/modules/specialists/builtin/*.py once."""
    if _BUILTINS:
        return dict(_BUILTINS)
    try:
        from . import builtin as builtin_pkg
    except Exception:
        logger.exception('specialist.builtin_import_failed')
        return {}
    for module in pkgutil.iter_modules(builtin_pkg.__path__):
        if module.name.startswith('_'):
            continue
        try:
            __import__(f'{builtin_pkg.__name__}.{module.name}', fromlist=['*'])
        except Exception:
            logger.exception('specialist.builtin_load_failed | module=%s', module.name)
    return dict(_BUILTINS)


def get_builtin(key: str) -> Optional[SpecialistDef]:
    return _BUILTINS.get(key)
