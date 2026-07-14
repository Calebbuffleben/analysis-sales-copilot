"""Tests for acoustic fingerprint extraction."""

from src.modules.acoustic_fingerprint.fingerprint_generator import (
    FingerprintGenerator,
    compute_energy_dbfs,
    pcm16_to_float32,
)
from src.modules.acoustic_fingerprint.synthetic_corpus import build_self_roundtrip_session


def test_fingerprint_generator_emits_vectors_for_speech_like_audio() -> None:
    session = build_self_roundtrip_session()
    generator = FingerprintGenerator()
    fingerprints = generator.fingerprint_stream(
        session.mic_pcm,
        user_id=session.seller_user_id,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
    )
    assert fingerprints
    assert all(len(fp.features) > 0 for fp in fingerprints)
    assert fingerprints[0].feature_type == 'logmel_mfcc_v1'


def test_silence_is_filtered_by_energy_threshold() -> None:
    generator = FingerprintGenerator()
    silence = pcm16_to_float32(b'\x00\x00' * 3200)
    assert compute_energy_dbfs(silence) <= -50.0
    fp = generator.fingerprint_from_window(
        silence,
        user_id='seller-a',
        seller_room_id='room-1',
        meeting_id='meet-1',
        seq=0,
        capture_time_ms=0,
    )
    assert fp is None
