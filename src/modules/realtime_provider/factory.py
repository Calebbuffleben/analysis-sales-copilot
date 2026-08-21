"""Factory: LIVE_PROVIDER=gemini (default). Add a module + env to swap vendors."""

from __future__ import annotations

from .gemini_live_provider import GeminiLiveProvider
from .protocol import RealtimeCoachProvider

_PROVIDERS = {
    'gemini': GeminiLiveProvider,
}


def create_realtime_provider(
    *,
    name: str,
    api_key: str,
    model_name: str,
    context_window_tokens: int = 12_000,
) -> RealtimeCoachProvider:
    key = (name or 'gemini').strip().lower() or 'gemini'
    cls = _PROVIDERS.get(key)
    if cls is None:
        known = ', '.join(sorted(_PROVIDERS))
        raise ValueError(f'Unknown LIVE_PROVIDER={name!r}. Known: {known}')
    return cls(
        api_key=api_key,
        model_name=model_name,
        context_window_tokens=context_window_tokens,
    )
