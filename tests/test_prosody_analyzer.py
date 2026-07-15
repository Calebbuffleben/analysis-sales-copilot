"""Tests for lightweight turn prosody enrichment."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock

from src.modules.audio_buffer.prosody_analyzer import (
    ProsodySnapshot,
    analyze_turn_prosody,
)
from src.modules.text_analysis.live_feedback_publisher import LiveFeedbackPublisher


def _pcm_tone(samples: int, amplitude: int = 5000) -> bytes:
    return b''.join(struct.pack('<h', amplitude) for _ in range(samples))


def _pcm_silence(samples: int) -> bytes:
    return b'\x00\x00' * samples


def test_silence_snapshot_is_emptyish() -> None:
    snap = analyze_turn_prosody(_pcm_silence(1600), sample_rate=16000, channels=1)
    assert snap.samples_count == 1600
    assert snap.speech_count == 0
    assert snap.hesitation_hint == 'none'
    assert snap.nudge_line() == ''


def test_continuous_speech_no_internal_pause() -> None:
    # 1s of continuous tone — no internal silence gaps.
    snap = analyze_turn_prosody(_pcm_tone(16000), sample_rate=16000, channels=1)
    assert snap.samples_count == 16000
    assert snap.speech_ratio > 0.9
    assert snap.pause_count == 0
    assert snap.longest_pause_ms == 0
    assert snap.hesitation_hint == 'none'
    assert snap.mean_rms_dbfs is not None


def test_speech_with_long_internal_pause_detects_hesitation() -> None:
    # 400ms speech + 900ms silence + 400ms speech (internal pause ~900ms).
    pcm = _pcm_tone(6400) + _pcm_silence(14400) + _pcm_tone(6400)
    snap = analyze_turn_prosody(pcm, sample_rate=16000, channels=1)
    assert snap.pause_count >= 1
    assert snap.longest_pause_ms >= 800
    assert snap.hesitation_hint == 'moderate'
    assert snap.is_distinctive()
    line = snap.nudge_line()
    assert line.startswith('Prosódia turno anterior:')
    assert 'pausa' in line or 'hesitação' in line or 'energia' in line


def test_publisher_merges_prosody_stats() -> None:
    dispatcher = MagicMock()
    dispatcher.enqueue.return_value = True
    publisher = LiveFeedbackPublisher(dispatcher, min_confidence=0.6)
    prosody = ProsodySnapshot(
        samples_count=1000,
        speech_count=800,
        mean_rms_dbfs=-25.5,
        speech_ratio=0.8,
        duration_ms=1000,
        pause_count=1,
        longest_pause_ms=450,
        internal_pause_ratio=0.2,
        energy_level='mid',
        hesitation_hint='weak',
        energy_variance=1.0,
    )
    ok = publisher.publish_tool_call(
        meeting_id='m1',
        tenant_id='t1',
        participant_id='p1',
        participant_role='client',
        args={
            'turnId': 'turn-1',
            'feedback': 'Pergunte o orçamento.',
            'confidence': 0.9,
            'feedback_type': 'opportunity',
            'evidence_text': 'quanto custa',
            'estado': {},
        },
        speech_end_ms=1_700_000_000_000,
        turn_id='turn-1',
        prosody=prosody,
    )
    assert ok is True
    event = dispatcher.enqueue.call_args[0][0]
    assert event.analysis.samples_count == 1000
    assert event.analysis.speech_count == 800
    assert event.analysis.mean_rms_dbfs == -25.5
    parsed = json.loads(event.analysis.prosody_json)
    assert parsed['hesitation_hint'] == 'weak'
    assert parsed['longest_pause_ms'] == 450


def test_publisher_works_without_prosody() -> None:
    dispatcher = MagicMock()
    dispatcher.enqueue.return_value = True
    publisher = LiveFeedbackPublisher(dispatcher, min_confidence=0.6)
    ok = publisher.publish_tool_call(
        meeting_id='m1',
        tenant_id='t1',
        participant_id='p1',
        participant_role='client',
        args={
            'turnId': 'turn-2',
            'feedback': 'Confirme o próximo passo.',
            'confidence': 0.85,
            'feedback_type': 'closing',
            'evidence_text': 'vamos seguir',
            'estado': {},
        },
        speech_end_ms=1_700_000_000_100,
        turn_id='turn-2',
        prosody=None,
    )
    assert ok is True
    event = dispatcher.enqueue.call_args[0][0]
    assert event.analysis.samples_count is None
    assert event.analysis.prosody_json == ''
