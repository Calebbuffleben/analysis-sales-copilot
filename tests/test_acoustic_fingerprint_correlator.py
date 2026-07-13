"""Tests for loopback correlation against remote seller fingerprints."""

from src.modules.acoustic_fingerprint.fingerprint_correlator import FingerprintCorrelator
from src.modules.acoustic_fingerprint.fingerprint_generator import FingerprintGenerator
from src.modules.acoustic_fingerprint.synthetic_corpus import (
    build_customer_only_session,
    build_self_roundtrip_session,
)


def test_self_roundtrip_prefers_seller_classification() -> None:
    session = build_self_roundtrip_session(lag_ms=200)
    generator = FingerprintGenerator()
    correlator = FingerprintCorrelator(generator=generator)
    remote = generator.fingerprint_stream(
        session.mic_pcm,
        user_id=session.seller_user_id,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
    )
    results = correlator.correlate_stream(
        session.loopback_pcm,
        remote_fingerprints=remote,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
        simulated_lag_ms=session.simulated_lag_ms,
    )
    seller_hits = sum(1 for result in results if result.acoustic_class == 'seller')
    assert seller_hits >= max(1, len(results) // 4)


def test_customer_only_does_not_force_seller_match() -> None:
    session = build_customer_only_session()
    generator = FingerprintGenerator()
    correlator = FingerprintCorrelator(generator=generator)
    remote = generator.fingerprint_stream(
        session.mic_pcm,
        user_id=session.seller_user_id,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
    )
    results = correlator.correlate_stream(
        session.loopback_pcm,
        remote_fingerprints=remote,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
    )
    seller_hits = sum(1 for result in results if result.acoustic_class == 'seller')
    assert seller_hits <= max(1, len(results) // 5)
