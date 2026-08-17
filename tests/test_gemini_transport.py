"""Tests for Gemini multi-transport auth chain."""

from unittest.mock import MagicMock, patch

import pytest

from src.modules.text_analysis.gemini_transport import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_THINKING_BUDGET,
    GeminiTransportMode,
    generate_with_transport_chain,
    is_auth_error_message,
    rest_generation_config,
    transport_candidates,
)


def test_aq_keys_use_vertex_express_transports() -> None:
    modes = transport_candidates('AQ.test-key')
    assert modes == (
        GeminiTransportMode.SDK_VERTEX_EXPRESS,
        GeminiTransportMode.REST_VERTEX_HEADER,
        GeminiTransportMode.REST_VERTEX_QUERY,
    )
    assert GeminiTransportMode.SDK_DEVELOPER not in modes


def test_aiza_keys_use_developer_transports() -> None:
    modes = transport_candidates('AIzaSyExample')
    assert GeminiTransportMode.SDK_DEVELOPER in modes
    assert GeminiTransportMode.SDK_VERTEX_EXPRESS not in modes


def test_is_auth_error_message() -> None:
    assert is_auth_error_message('401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED')
    assert not is_auth_error_message('timeout')


def test_rest_generation_config_disables_thinking() -> None:
    config = rest_generation_config(json_mode=True)
    assert config['maxOutputTokens'] == GEMINI_MAX_OUTPUT_TOKENS
    assert config['thinkingConfig'] == {'thinkingBudget': GEMINI_THINKING_BUDGET}
    assert config['responseMimeType'] == 'application/json'


def test_sdk_generation_config_disables_thinking() -> None:
    fake_types = MagicMock()
    fake_thinking = MagicMock()
    fake_config = MagicMock()
    fake_types.ThinkingConfig = fake_thinking
    fake_types.GenerateContentConfig = fake_config

    with patch.dict('sys.modules', {'google.genai': MagicMock(types=fake_types)}):
        from src.modules.text_analysis.gemini_transport import sdk_generation_config

        sdk_generation_config(json_mode=True)

    fake_thinking.assert_called_once_with(thinking_budget=GEMINI_THINKING_BUDGET)
    fake_config.assert_called_once()
    _, kwargs = fake_config.call_args
    assert kwargs['max_output_tokens'] == GEMINI_MAX_OUTPUT_TOKENS
    assert kwargs['response_mime_type'] == 'application/json'


def test_generate_with_transport_chain_falls_back_on_auth_error() -> None:
    calls: list[GeminiTransportMode] = []

    def fake_generate(**kwargs):
        mode = kwargs['mode']
        calls.append(mode)
        if mode == GeminiTransportMode.SDK_VERTEX_EXPRESS:
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
    assert mode == GeminiTransportMode.REST_VERTEX_HEADER
    assert calls[0] == GeminiTransportMode.SDK_VERTEX_EXPRESS
    assert calls[1] == GeminiTransportMode.REST_VERTEX_HEADER


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
