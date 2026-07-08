"""Tests for Gemini analyzer transport integration."""

from unittest.mock import MagicMock, patch
from src.modules.text_analysis.gemini_analyzer import (
    GeminiAnalyzer,
    uses_vertex_express_api,
)
from src.modules.text_analysis.gemini_transport import GeminiTransportMode


def test_uses_vertex_express_api_for_aq_prefix() -> None:
    assert uses_vertex_express_api('AQ.Ab8RN6Example') is True
    assert uses_vertex_express_api('AIzaSyExample') is False


def test_analyzer_uses_transport_chain_for_aq_keys() -> None:
    analyzer = GeminiAnalyzer(
        api_key='AQ.test-key',
        model_name='gemini-2.5-flash',
        slot_index=2,
    )

    with patch(
        'src.modules.text_analysis.gemini_analyzer.generate_with_transport_chain',
        return_value=(
            '{"feedback": null, "confidence": 0.0, "feedback_type": null, "estado": {}}',
            GeminiTransportMode.REST_DEVELOPER_HEADER,
        ),
    ) as chain:
        result = analyzer.analyze('objeção de preço', {}, speaker_role='host')

    chain.assert_called_once()
    assert result['direct_feedback'] == ''
    assert analyzer._cached_transport == GeminiTransportMode.REST_DEVELOPER_HEADER


def test_analyzer_sdk_injection_path() -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(
        text='{"feedback": null, "confidence": 0.0, "feedback_type": null, "estado": {}}',
    )
    fake_types = MagicMock()
    with patch(
        'src.modules.text_analysis.gemini_analyzer.sdk_generation_config',
        return_value=fake_types,
    ) as config_builder:
        analyzer = GeminiAnalyzer(
            api_key='AIzaSyExample',
            client=fake_client,
        )
        result = analyzer.analyze('teste', {}, speaker_role='host')

    assert result['direct_feedback'] == ''
    config_builder.assert_called_once_with(json_mode=True)
    fake_client.models.generate_content.assert_called_once()
    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs['config'] is fake_types
