"""Unit tests for PCM v2 and turn aggregation."""

from src.modules.acoustic_fingerprint.pcm_v2 import (
    AcousticWindowLabel,
    aggregate_turn_acoustic_class,
    is_pcm_v2,
    parse_label_control,
    try_decode_pcm_v2,
)
import struct


def test_pcm_v2_roundtrip_header() -> None:
    magic = 0x4D503206
    pcm = b'\x00\x01' * 160
    header = struct.pack('>IIIIII', magic, 7, 1000, 3, len(pcm), 0)
    frame = header + pcm
    assert is_pcm_v2(frame)
    decoded = try_decode_pcm_v2(frame)
    assert decoded is not None
    assert decoded.frame_seq == 7
    assert decoded.label_id == 3
    assert decoded.pcm == pcm


def test_parse_label_control() -> None:
    label = parse_label_control(
        '{"type":"acoustic_label","labelId":2,"acousticClass":"seller","confidence":0.9,"lagMs":120,"windowStartMs":0,"windowEndMs":200}',
    )
    assert label is not None
    assert label.acoustic_class == 'seller'
    assert label.confidence == 0.9


def test_aggregate_turn_prefers_seller_with_margin() -> None:
    labels = [
        AcousticWindowLabel(1, 'seller', 'u1', 0.9, 0, 0, 200),
        AcousticWindowLabel(2, 'seller', 'u1', 0.85, 0, 200, 400),
        AcousticWindowLabel(3, 'customer', None, 0.5, 0, 400, 500),
    ]
    assert aggregate_turn_acoustic_class(labels, 0, 500) == 'seller'
