"""Turn-level acoustic aggregation used by AssemblyAI final turns."""

from src.modules.acoustic_fingerprint.pcm_v2 import (
    AcousticLabelBuffer,
    AcousticWindowLabel,
    aggregate_turn_acoustic_class,
)


def test_aggregate_prefers_seller_with_margin():
    labels = [
        AcousticWindowLabel(1, 'seller', 'u1', 0.9, 0, 0, 200),
        AcousticWindowLabel(2, 'seller', 'u1', 0.85, 0, 200, 400),
        AcousticWindowLabel(3, 'customer', None, 0.5, 0, 400, 500),
    ]
    assert aggregate_turn_acoustic_class(labels, 0, 500) == 'seller'


def test_label_buffer_aggregate_and_current():
    buf = AcousticLabelBuffer()
    buf.upsert(AcousticWindowLabel(1, 'customer', None, 0.9, 0, 0, 200))
    buf.upsert(AcousticWindowLabel(2, 'customer', None, 0.9, 0, 200, 400))
    assert buf.aggregate(0, 400) == 'customer'
    assert buf.current_class() == 'customer'


def test_unknown_when_insufficient_evidence():
    labels = [
        AcousticWindowLabel(1, 'seller', 'u1', 0.5, 0, 0, 200),
        AcousticWindowLabel(2, 'customer', None, 0.5, 0, 200, 400),
    ]
    assert aggregate_turn_acoustic_class(labels, 0, 400) == 'unknown'
