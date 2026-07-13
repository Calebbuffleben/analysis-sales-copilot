"""PCM v2 envelope parser for acoustic labels on the wire."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Literal

AcousticClass = Literal['seller', 'customer', 'unknown']

PCM_V2_MAGIC = 0x4D503206
PCM_V2_HEADER_BYTES = 24


@dataclass(frozen=True)
class AcousticWindowLabel:
    label_id: int
    acoustic_class: AcousticClass
    matched_seller_id: str | None
    confidence: float
    lag_ms: int
    window_start_ms: int
    window_end_ms: int


@dataclass(frozen=True)
class PcmV2Frame:
    frame_seq: int
    capture_mono_ms: int
    label_id: int
    pcm: bytes


def is_pcm_v2(data: bytes) -> bool:
    if len(data) < 4:
        return False
    magic = struct.unpack('>I', data[:4])[0]
    return magic == PCM_V2_MAGIC


def try_decode_pcm_v2(data: bytes) -> PcmV2Frame | None:
    if len(data) < PCM_V2_HEADER_BYTES:
        return None
    magic, frame_seq, capture_mono_ms, label_id, pcm_length, _reserved = struct.unpack(
        '>IIIIII',
        data[:PCM_V2_HEADER_BYTES],
    )
    if magic != PCM_V2_MAGIC:
        return None
    if len(data) < PCM_V2_HEADER_BYTES + pcm_length:
        return None
    pcm = data[PCM_V2_HEADER_BYTES : PCM_V2_HEADER_BYTES + pcm_length]
    return PcmV2Frame(
        frame_seq=frame_seq,
        capture_mono_ms=capture_mono_ms,
        label_id=label_id,
        pcm=pcm,
    )


def parse_label_control(text: str) -> AcousticWindowLabel | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get('type') != 'acoustic_label':
        return None
    acoustic_class = str(
        obj.get('acousticClass') or obj.get('acoustic_class') or 'unknown',
    )
    if acoustic_class not in {'seller', 'customer', 'unknown'}:
        acoustic_class = 'unknown'
    matched = obj.get('matched_seller_id')
    return AcousticWindowLabel(
        label_id=int(obj.get('labelId') or obj.get('label_id') or 0),
        acoustic_class=acoustic_class,  # type: ignore[arg-type]
        matched_seller_id=str(matched) if matched else None,
        confidence=float(obj.get('confidence') or 0.0),
        lag_ms=int(obj.get('lagMs') or obj.get('lag_ms') or 0),
        window_start_ms=int(obj.get('windowStartMs') or obj.get('window_start_ms') or 0),
        window_end_ms=int(obj.get('windowEndMs') or obj.get('window_end_ms') or 0),
    )


def aggregate_turn_acoustic_class(
    labels: list[AcousticWindowLabel],
    start_ms: int,
    end_ms: int,
) -> AcousticClass:
    seller_score = 0.0
    customer_score = 0.0
    total = 0.0
    for label in labels:
        overlap = min(end_ms, label.window_end_ms) - max(start_ms, label.window_start_ms)
        if overlap <= 0:
            continue
        weight = overlap * max(0.1, label.confidence)
        total += weight
        if label.acoustic_class == 'seller':
            seller_score += weight
        elif label.acoustic_class == 'customer':
            customer_score += weight
    if total <= 0:
        return 'unknown'
    s = seller_score / total
    c = customer_score / total
    if s >= 0.65 and s - c >= 0.15:
        return 'seller'
    if c >= 0.75 and s <= 0.2:
        return 'customer'
    return 'unknown'


class AcousticLabelBuffer:
    """Ring of recent acoustic labels keyed by label_id / time."""

    def __init__(self, max_labels: int = 200) -> None:
        self._labels: list[AcousticWindowLabel] = []
        self._by_id: dict[int, AcousticWindowLabel] = {}
        self._max = max_labels
        self._current: AcousticWindowLabel | None = None

    def upsert(self, label: AcousticWindowLabel) -> None:
        self._by_id[label.label_id] = label
        self._current = label
        self._labels.append(label)
        if len(self._labels) > self._max:
            old = self._labels.pop(0)
            if self._by_id.get(old.label_id) is old:
                self._by_id.pop(old.label_id, None)

    def resolve_for_label_id(self, label_id: int) -> AcousticWindowLabel | None:
        return self._by_id.get(label_id) or self._current

    def current_class(self) -> AcousticClass:
        if self._current is None:
            return 'unknown'
        return self._current.acoustic_class

    def aggregate(self, start_ms: int, end_ms: int) -> AcousticClass:
        return aggregate_turn_acoustic_class(self._labels, start_ms, end_ms)
