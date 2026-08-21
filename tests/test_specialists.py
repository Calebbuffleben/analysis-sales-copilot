"""Code specialist decorator + catalog merge."""

from src.modules.specialists.decorator import load_builtins, specialist
from src.modules.specialists.fanout import _matches_trigger
from src.modules.specialists.types import SpecialistDef, SpecialistTurnContext


def test_spin_coach_builtin_registers() -> None:
    builtins = load_builtins()
    assert 'spin_coach' in builtins
    spec = builtins['spin_coach']
    assert spec.source == 'code'
    assert spec.name


def test_decorator_adds_specialist() -> None:
    @specialist(key='unit_only', name='Unit', description='test')
    def _prompt(ctx, spec):
        return spec.key

    builtins = load_builtins()
    assert builtins['unit_only'].name == 'Unit'


def test_trigger_keywords() -> None:
    spec = SpecialistDef(
        key='price',
        name='Preço',
        trigger_keywords=('preço', 'caro'),
        trigger_phases=(),
    )
    ctx = SpecialistTurnContext(
        tenant_id='t',
        meeting_id='m',
        participant_id='p',
        participant_role='participant',
        turn_id='1',
        speech_end_ms=1,
        evidence_text='está caro demais',
        primary_feedback='',
        conversation_state={},
        host_context='',
    )
    assert _matches_trigger(spec, ctx) is True
    ctx.evidence_text = 'vamos seguir'
    assert _matches_trigger(spec, ctx) is False
