"""Tests for Gemini multi-transport auth chain."""

from unittest.mock import patch

import pytest

from src.modules.text_analysis.gemini_transport import (
    GeminiTransportMode,
    generate_with_transport_chain,
    is_auth_error_message,
    transport_candidates,
)


def test_aq_keys_try_all_transports() -> None:
    modes = transport_candidates('AQ.test-key')
    assert GeminiTransportMode.REST_DEVELOPER_HEADER in modes
    assert GeminiTransportMode.SDK_VERTEX_EXPRESS in modes
    assert modes[0] == GeminiTransportMode.REST_DEVELOPER_HEADER


def test_is_auth_error_message() -> None:
    assert is_auth_error_message('401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED')
    assert not is_auth_error_message('timeout')


def test_generate_with_transport_chain_falls_back_on_auth_error() -> None:
    calls: list[GeminiTransportMode] = []

    def fake_generate(**kwargs):
        mode = kwargs['mode']
        calls.append(mode)
        if mode == GeminiTransportMode.REST_DEVELOPER_HEADER:
            raise RuntimeError('401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED')
        return '{"feedback": null}'

    with patch(
        'src.modules.text_analysis.gemini_transport.generate_content_text',
        side_effect=fake_generate,
    ):
        text, mode = generate_with_transport_chain(
            api_key='AQ.test-key',
            model_name='gemini-2.5-flash',
            prompt='hello',
            slot_index=2,
        )

    assert text == '{"feedback": null}'
    assert mode == GeminiTransportMode.REST_DEVELOPER_QUERY
    assert calls[0] == GeminiTransportMode.REST_DEVELOPER_HEADER
    assert calls[1] == GeminiTransportMode.REST_DEVELOPER_QUERY


def test_generate_with_transport_chain_raises_when_all_fail() -> None:
    with patch(
        'src.modules.text_analysis.gemini_transport.generate_content_text',
        side_effect=RuntimeError('401 UNAUTHENTICATED'),
    ):
        with pytest.raises(RuntimeError):
            generate_with_transport_chain(
                api_key='AQ.test-key',
                model_name='gemini-2.5-flash',
                prompt='hello',
            )
