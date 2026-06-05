"""Tests for Gemini client factory (AIza vs AQ. API key formats)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.modules.text_analysis.gemini_analyzer import (
    GeminiAnalyzer,
    create_genai_client,
    uses_vertex_express_api,
)


@pytest.fixture
def mock_genai_module(monkeypatch):
    """Inject a fake google.genai module for lazy imports inside create_genai_client."""
    mock_genai = MagicMock()
    monkeypatch.setitem(sys.modules, 'google', types.SimpleNamespace(genai=mock_genai))
    return mock_genai


def test_uses_vertex_express_api_for_aq_prefix() -> None:
    assert uses_vertex_express_api('AQ.Ab8RN6Example') is True
    assert uses_vertex_express_api('AIzaSyExample') is False


def test_create_genai_client_legacy_aiza_key(mock_genai_module) -> None:
    client = create_genai_client('AIzaSyExampleKey', slot_index=0)
    mock_genai_module.Client.assert_called_once_with(
        api_key='AIzaSyExampleKey',
        vertexai=False,
    )
    assert client is mock_genai_module.Client.return_value


def test_create_genai_client_aq_key_returns_none_for_rest_transport() -> None:
    assert create_genai_client('AQ.Ab8RN6Example', slot_index=2) is None


def test_gemini_analyzer_aq_key_uses_rest_transport() -> None:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._api_key = 'AQ.test-key'
    analyzer._use_rest_transport = True
    analyzer.model_name = 'gemini-2.5-flash'

    with patch.object(
        analyzer,
        '_rest_generate_content',
        return_value='{"feedback": null, "confidence": 0.0, "feedback_type": null, "estado": {}}',
    ) as rest_call:
        from src.modules.text_analysis.llm_state_validator import validate_llm_response

        with patch(
            'src.modules.text_analysis.gemini_analyzer.validate_llm_response',
            wraps=validate_llm_response,
        ):
            analyzer._consecutive_429_errors = 0
            analyzer._backoff_until_ms = 0
            analyzer._slot_index = 1
            analyzer._api_key_prefix = 'AQ.test-...'
            analyzer._generation_config = lambda **kwargs: kwargs
            result = analyzer.analyze('teste', {}, speaker_role='host')

    rest_call.assert_called_once()
    assert result['direct_feedback'] == ''


def test_rest_generate_content_uses_vertex_express_endpoint() -> None:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._api_key = 'AQ.secret-key'
    analyzer.model_name = 'gemini-2.5-flash'
    analyzer._slot_index = 2

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        'candidates': [{'content': {'parts': [{'text': '{"feedback": null}'}]}}],
    }
    mock_response.raise_for_status = MagicMock()

    with patch('requests.post', return_value=mock_response) as post:
        text = analyzer._rest_generate_content('prompt')

    assert text == '{"feedback": null}'
    post.assert_called_once()
    url = post.call_args[0][0]
    kwargs = post.call_args[1]
    assert 'aiplatform.googleapis.com/v1/publishers/google/models/' in url
    assert kwargs['params']['key'] == 'AQ.secret-key'
    assert 'Authorization' not in kwargs['headers']
