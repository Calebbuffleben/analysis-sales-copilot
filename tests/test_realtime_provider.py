"""Factory + GeminiCoachSession contracts for swapping Live vendors."""

from src.modules.realtime_provider import create_realtime_provider
from src.modules.realtime_provider.gemini_live_provider import (
    EMIT_FEEDBACK_TOOL,
    GeminiCoachSession,
    GeminiLiveProvider,
)
from src.modules.realtime_provider.types import (
    CoachTurnResult,
    turn_result_from_tool_args,
)


def test_factory_rejects_unknown_provider() -> None:
    try:
        create_realtime_provider(
            name='not-a-vendor',
            api_key='x',
            model_name='m',
        )
    except ValueError as exc:
        assert 'LIVE_PROVIDER' in str(exc)
        return
    raise AssertionError('expected ValueError')


def test_factory_returns_gemini() -> None:
    provider = create_realtime_provider(
        name='gemini',
        api_key='AIzaSyTest',
        model_name='gemini-3.1-flash-live-preview',
    )
    assert isinstance(provider, GeminiLiveProvider)
    assert provider.name == 'gemini'
    assert 'emit_feedback' in EMIT_FEEDBACK_TOOL['function_declarations'][0]['name']


def test_turn_result_from_tool_args() -> None:
    result = turn_result_from_tool_args(
        {
            'turnId': 't1',
            'feedback': 'Pergunte o impacto',
            'confidence': 0.9,
            'feedback_type': 'opportunity',
            'evidence_text': 'isso atrasou o trimestre',
            'estado': {'fase_spin': 'implicacao'},
            'playbook_template_key': 'spin',
            'playbook_variables': {'q': 'impacto'},
        },
    )
    assert isinstance(result, CoachTurnResult)
    assert result.turn_id == 't1'
    assert result.playbook_template_key == 'spin'
    assert result.conversation_state['fase_spin'] == 'implicacao'


def test_parse_tool_calls_from_mock_response() -> None:
    class Fc:
        name = 'emit_feedback'
        args = {'turnId': 'abc', 'feedback': ''}
        id = 'fc-1'

    class ToolCall:
        function_calls = [Fc()]

    class Response:
        tool_call = ToolCall()

    parsed = GeminiCoachSession.parse_tool_calls(Response())
    assert len(parsed) == 1
    name, args, fc = parsed[0]
    assert name == 'emit_feedback'
    assert args['turnId'] == 'abc'
    assert fc.id == 'fc-1'
