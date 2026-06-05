"""Gemini API transport: probes and generates across Developer + Express endpoints."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SEC = 60.0


class GeminiTransportMode(str, Enum):
    SDK_DEVELOPER = 'sdk_developer'
    SDK_VERTEX_EXPRESS = 'sdk_vertex_express'
    REST_DEVELOPER_HEADER = 'rest_developer_header'
    REST_DEVELOPER_QUERY = 'rest_developer_query'
    REST_VERTEX_HEADER = 'rest_vertex_header'
    REST_VERTEX_QUERY = 'rest_vertex_query'


def key_prefix(api_key: str) -> str:
    normalized = (api_key or '').strip()
    if len(normalized) <= 8:
        return 'unset'
    return normalized[:8] + '...'


def transport_candidates(api_key: str) -> tuple[GeminiTransportMode, ...]:
    """Ordered transports to try. AQ keys get every path; AIza keys skip Vertex SDK."""
    normalized = (api_key or '').strip()
    if normalized.startswith('AQ.'):
        return (
            GeminiTransportMode.REST_DEVELOPER_HEADER,
            GeminiTransportMode.REST_DEVELOPER_QUERY,
            GeminiTransportMode.SDK_DEVELOPER,
            GeminiTransportMode.REST_VERTEX_HEADER,
            GeminiTransportMode.REST_VERTEX_QUERY,
            GeminiTransportMode.SDK_VERTEX_EXPRESS,
        )
    return (
        GeminiTransportMode.SDK_DEVELOPER,
        GeminiTransportMode.REST_DEVELOPER_HEADER,
        GeminiTransportMode.REST_DEVELOPER_QUERY,
    )


def is_auth_error_message(message: str) -> bool:
    normalized = message or ''
    return any(
        token in normalized
        for token in (
            '401',
            'UNAUTHENTICATED',
            'ACCESS_TOKEN_TYPE_UNSUPPORTED',
            'API_KEY_INVALID',
            'API key not valid',
        )
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get('candidates') or []
    if not candidates:
        return ''
    parts = (candidates[0].get('content') or {}).get('parts') or []
    texts = [part.get('text', '') for part in parts if isinstance(part, dict)]
    return ''.join(texts).strip()


def _rest_request(
    *,
    mode: GeminiTransportMode,
    api_key: str,
    model_name: str,
    prompt: str,
    json_mode: bool,
) -> str:
    generation_config: dict[str, Any] = {'temperature': 0.2}
    if json_mode:
        generation_config['responseMimeType'] = 'application/json'

    body = {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': generation_config,
    }

    if mode in (
        GeminiTransportMode.REST_DEVELOPER_HEADER,
        GeminiTransportMode.REST_DEVELOPER_QUERY,
    ):
        base = 'https://generativelanguage.googleapis.com/v1beta/models/'
    else:
        base = 'https://aiplatform.googleapis.com/v1/publishers/google/models/'

    url = f'{base}{model_name}:generateContent'
    headers = {'Content-Type': 'application/json'}
    params: dict[str, str] | None = None

    if mode in (
        GeminiTransportMode.REST_DEVELOPER_HEADER,
        GeminiTransportMode.REST_VERTEX_HEADER,
    ):
        headers['x-goog-api-key'] = api_key
    else:
        params = {'key': api_key}

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=body,
        timeout=_REQUEST_TIMEOUT_SEC,
    )
    if not response.ok:
        logger.warning(
            'Gemini REST failed | transport=%s | prefix=%s | status=%s | body=%s',
            mode.value,
            key_prefix(api_key),
            response.status_code,
            response.text[:300],
        )
        response.raise_for_status()
    return _extract_response_text(response.json())


def _sdk_request(
    *,
    mode: GeminiTransportMode,
    api_key: str,
    model_name: str,
    prompt: str,
    json_mode: bool,
) -> str:
    from google import genai
    from google.genai import types

    vertexai = mode == GeminiTransportMode.SDK_VERTEX_EXPRESS
    client = genai.Client(api_key=api_key, vertexai=vertexai)
    config_kwargs: dict[str, Any] = {'temperature': 0.2}
    if json_mode:
        config_kwargs['response_mime_type'] = 'application/json'
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or '').strip()


def generate_content_text(
    *,
    mode: GeminiTransportMode,
    api_key: str,
    model_name: str,
    prompt: str,
    json_mode: bool = True,
) -> str:
    if mode in (GeminiTransportMode.SDK_DEVELOPER, GeminiTransportMode.SDK_VERTEX_EXPRESS):
        return _sdk_request(
            mode=mode,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            json_mode=json_mode,
        )
    return _rest_request(
        mode=mode,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        json_mode=json_mode,
    )


def generate_with_transport_chain(
    *,
    api_key: str,
    model_name: str,
    prompt: str,
    preferred_mode: Optional[GeminiTransportMode] = None,
    json_mode: bool = True,
    slot_index: Optional[int] = None,
) -> tuple[str, GeminiTransportMode]:
    """Try transports in order until one succeeds. Returns text + winning mode."""
    candidates: list[GeminiTransportMode] = []
    if preferred_mode is not None:
        candidates.append(preferred_mode)
    for mode in transport_candidates(api_key):
        if mode not in candidates:
            candidates.append(mode)

    last_error: Optional[Exception] = None
    for mode in candidates:
        try:
            text = generate_content_text(
                mode=mode,
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                json_mode=json_mode,
            )
            if preferred_mode != mode:
                logger.info(
                    'Gemini transport selected | slot=%s | prefix=%s | transport=%s',
                    slot_index,
                    key_prefix(api_key),
                    mode.value,
                )
            return text, mode
        except Exception as exc:
            if is_auth_error_message(str(exc)):
                last_error = exc
                continue
            raise

    raise last_error or RuntimeError(
        f'No Gemini transport succeeded for slot {slot_index}',
    )
