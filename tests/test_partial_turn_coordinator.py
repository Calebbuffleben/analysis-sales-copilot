"""Tests for PartialTurnCoordinator stability and completeness gating."""

from __future__ import annotations

from src.modules.text_analysis.types import TranscriptionChunk
from src.modules.transcription.partial_turn_coordinator import (
    FinalAction,
    PartialTurnConfig,
    PartialTurnCoordinator,
)


def _participant_meta() -> dict[str, object]:
    return {
        'meeting_id': 'meet-1',
        'participant_id': 'meet-remote',
        'track': 'tab-audio',
        'participant_role': 'participant',
        'tenant_id': 'tenant-1',
        'transcript_confidence': 0.9,
        'turn_start_ms': 1_000,
    }


def test_unstable_partial_not_dispatched() -> None:
    dispatched: list[tuple[str, TranscriptionChunk, dict[str, object]]] = []

    coordinator = PartialTurnCoordinator(
        PartialTurnConfig(stable_ms=600, word_stable_ms=300),
        lambda sk, chunk, extra: dispatched.append((sk, chunk, extra)),
    )

    coordinator.handle_partial(
        'meet-1:meet-remote:tab-audio',
        'Achei muito caro comparado ao concorrente.',
        5_000,
        _participant_meta(),
    )
    assert dispatched == []


def test_stable_complete_partial_dispatched() -> None:
    dispatched: list[tuple[str, TranscriptionChunk, dict[str, object]]] = []

    coordinator = PartialTurnCoordinator(
        PartialTurnConfig(
            stable_ms=100,
            word_stable_ms=50,
            growth_window_ms=50,
            min_words=5,
            cooldown_ms=0,
        ),
        lambda sk, chunk, extra: dispatched.append((sk, chunk, extra)),
    )

    text = 'Achei muito caro comparado ao concorrente.'
    stream_key = 'meet-1:meet-remote:tab-audio'
    coordinator.handle_partial(stream_key, text, 1_000, _participant_meta())
    coordinator.handle_partial(stream_key, text, 1_200, _participant_meta())

    assert len(dispatched) == 1
    assert dispatched[0][1].text == text
    assert dispatched[0][2]['transcriptSource'] == 'partial'


def test_host_stream_ignored() -> None:
    dispatched: list[object] = []
    coordinator = PartialTurnCoordinator(
        PartialTurnConfig(stable_ms=50, word_stable_ms=25, cooldown_ms=0),
        lambda *_args: dispatched.append(_args),
    )
    coordinator.handle_partial(
        'meet-1:host:microphone',
        'A nossa solução reduz custos.',
        2_000,
        {
            'meeting_id': 'meet-1',
            'participant_id': 'host',
            'track': 'microphone',
            'participant_role': 'host',
        },
    )
    assert dispatched == []


def test_plan_final_skip_when_partial_published() -> None:
    coordinator = PartialTurnCoordinator(
        PartialTurnConfig(),
        lambda *_args: None,
    )
    text = 'Achei muito caro comparado ao concorrente.'
    coordinator.note_partial_published('stream-1', text, 0.85)
    assert coordinator.plan_final_action('stream-1', text) == FinalAction.SKIP


def test_plan_final_reconcile_when_text_diverged() -> None:
    coordinator = PartialTurnCoordinator(
        PartialTurnConfig(),
        lambda *_args: None,
    )
    coordinator.note_partial_published(
        'stream-1',
        'Achei muito caro comparado ao concorrente.',
        0.85,
    )
    action = coordinator.plan_final_action(
        'stream-1',
        'Achei muito caro comparado ao concorrente hoje.',
    )
    assert action == FinalAction.RECONCILE
