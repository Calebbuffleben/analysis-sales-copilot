"""PCM/WAV helpers for the acoustic feasibility harness."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .fingerprint_generator import pcm16_to_float32
from .types import CorpusSession, LabeledWindow


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints.tobytes()


def read_wav_mono(path: Path, *, target_rate: int = 16000) -> tuple[bytes, int]:
    data, sample_rate = sf.read(str(path), always_2d=True)
    mono = data[:, 0].astype(np.float32)
    if sample_rate != target_rate and mono.size > 0:
        duration = mono.size / sample_rate
        target_len = max(1, int(round(duration * target_rate)))
        x_old = np.linspace(0.0, 1.0, mono.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
        sample_rate = target_rate
    return float32_to_pcm16(mono), sample_rate


def write_wav_mono(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = pcm16_to_float32(pcm, 1)
    sf.write(str(path), samples, sample_rate, subtype='PCM_16')


def load_corpus_session(session_dir: Path) -> CorpusSession:
    manifest_path = session_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    mic_pcm, sample_rate = read_wav_mono(session_dir / 'mic.wav')
    loopback_pcm, _ = read_wav_mono(session_dir / 'loopback.wav', target_rate=sample_rate)
    labels = [
        LabeledWindow(
            start_ms=int(item['start_ms']),
            end_ms=int(item['end_ms']),
            ground_truth=item['ground_truth'],
            matched_seller_id=item.get('matched_seller_id'),
        )
        for item in manifest.get('labels', [])
    ]
    return CorpusSession(
        session_id=manifest['session_id'],
        scenario=manifest['scenario'],
        seller_user_id=manifest['seller_user_id'],
        listener_user_id=manifest['listener_user_id'],
        meeting_id=manifest['meeting_id'],
        seller_room_id=manifest['seller_room_id'],
        mic_pcm=mic_pcm,
        loopback_pcm=loopback_pcm,
        sample_rate=sample_rate,
        channels=1,
        labels=labels,
        simulated_lag_ms=int(manifest.get('simulated_lag_ms', 0)),
    )


def save_corpus_session(session: CorpusSession, output_dir: Path) -> None:
    session_dir = output_dir / session.session_id
    write_wav_mono(session_dir / 'mic.wav', session.mic_pcm, session.sample_rate)
    write_wav_mono(session_dir / 'loopback.wav', session.loopback_pcm, session.sample_rate)
    manifest = {
        'session_id': session.session_id,
        'scenario': session.scenario,
        'seller_user_id': session.seller_user_id,
        'listener_user_id': session.listener_user_id,
        'meeting_id': session.meeting_id,
        'seller_room_id': session.seller_room_id,
        'sample_rate': session.sample_rate,
        'channels': session.channels,
        'simulated_lag_ms': session.simulated_lag_ms,
        'labels': [asdict(label) for label in session.labels],
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
