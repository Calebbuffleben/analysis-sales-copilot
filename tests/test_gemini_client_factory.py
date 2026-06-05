"""Tests for Gemini client factory (AIza vs AQ. API key formats)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.modules.text_analysis.gemini_analyzer import create_genai_client


@pytest.fixture
def mock_genai_module(monkeypatch):
    """Inject a fake google.genai module for lazy imports inside create_genai_client."""
    mock_genai = MagicMock()
    monkeypatch.setitem(sys.modules, 'google', types.SimpleNamespace(genai=mock_genai))
    return mock_genai


def test_create_genai_client_legacy_aiza_key(mock_genai_module) -> None:
    create_genai_client('AIzaSyExampleKey', slot_index=0)
    mock_genai_module.Client.assert_called_once_with(api_key='AIzaSyExampleKey')


def test_create_genai_client_aq_key_uses_vertex_express(mock_genai_module) -> None:
    create_genai_client('AQ.Ab8RN6Example', slot_index=2)
    mock_genai_module.Client.assert_called_once_with(
        api_key='AQ.Ab8RN6Example',
        vertexai=True,
    )


def test_gemini_analyzer_delegates_to_factory() -> None:
    fake_client = MagicMock()
    fake_types = MagicMock()
    with patch(
        'src.modules.text_analysis.gemini_analyzer.create_genai_client',
        return_value=fake_client,
    ) as factory:
        with patch.dict(
            sys.modules,
            {
                'google': types.SimpleNamespace(genai=MagicMock()),
                'google.genai': types.SimpleNamespace(types=fake_types),
            },
        ):
            from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer

            analyzer = GeminiAnalyzer(api_key='AQ.test-key', slot_index=1)

    factory.assert_called_once_with('AQ.test-key', slot_index=1)
    assert analyzer.client is fake_client
