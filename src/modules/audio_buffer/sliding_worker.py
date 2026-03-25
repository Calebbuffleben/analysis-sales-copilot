"""SlidingWindowWorker orchestrates when to process sliding-window audio.

While CircularBuffer is responsible for storing the most recent PCM bytes
for a stream (the window), SlidingWindowWorker is responsible for deciding
*when* that window should be consumed by downstream processing.

Key ideas:
- It is notified every time new PCM is appended to a stream's buffer.
- It applies policy (e.g., only act when there is at least X seconds of
  audio, or at most once every Y seconds).
- When conditions are met, it reads the current window from the buffer
  (via AudioBufferService) and invokes a registered callback.

In the chosen architecture (opção B):
- SlidingWindowWorker não chama serviços de STT/NLP diretamente.
- Em vez disso, um serviço de nível superior, TranscriptionPipelineService
  (em modules/transcription/transcription_pipeline_service.py), registra
  um callback aqui e orquestra STT + análise de texto localmente no
  python-service (sem falar com o backend).

Planned responsibilities:
- Track per-stream timing/conditions para quando uma janela está "pronta".
- Interact with AudioBufferService (e indiretamente com CircularBuffer)
  para ler a janela PCM atual.
- Notificar o callback registrado quando uma janela estiver pronta, passando:
  (stream_key, window_pcm_bytes, meta_dict).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

from .audio_diagnostics import compute_pcm_window_stats

logger = logging.getLogger(__name__)

WindowDataReader = Callable[[str], Optional[Tuple[bytes, Dict[str, object]]]]
WindowReadyCallback = Callable[[str, bytes, Dict[str, object]], None]


class SlidingWindowWorker:
    """Apply ready-window policies and invoke the registered callback."""

    def __init__(
        self,
        window_reader: Optional[WindowDataReader] = None,
        min_window_seconds: float = 4.0,
        min_interval_ms: int = 2000,
    ) -> None:
        self._window_reader = window_reader
        self._window_callback: Optional[WindowReadyCallback] = None
        self._min_window_seconds = min_window_seconds
        self._min_interval_ms = min_interval_ms
        self._last_emitted_at_ms: Dict[str, int] = {}

    def set_window_reader(self, reader: WindowDataReader) -> None:
        """Register a callable that can return the latest window for a stream."""
        self._window_reader = reader

    def register_window_callback(self, callback: WindowReadyCallback) -> None:
        """Register the callback that consumes ready windows."""
        self._window_callback = callback

    def on_chunk_appended(self, stream_key: str, timestamp_ms: int) -> bool:
        """Try to emit a ready window for a stream after a new chunk arrives."""
        return self._emit_if_ready(stream_key, timestamp_ms, force=False)

    def flush_stream(self, stream_key: str) -> bool:
        """Force the current window to be emitted on stream teardown."""
        last_timestamp_ms = self._last_emitted_at_ms.get(stream_key, 0)
        return self._emit_if_ready(stream_key, last_timestamp_ms, force=True)

    def clear_stream_state(self, stream_key: str) -> None:
        """Drop per-stream worker state."""
        self._last_emitted_at_ms.pop(stream_key, None)

    def _emit_if_ready(
        self,
        stream_key: str,
        timestamp_ms: int,
        force: bool,
    ) -> bool:
        if self._window_reader is None or self._window_callback is None:
            return False

        window_data = self._window_reader(stream_key)
        if not window_data:
            return False

        window_pcm, meta = window_data
        if not window_pcm:
            return False

        bytes_per_second = (
            int(meta['sample_rate']) * int(meta['channels']) * 2
        )
        window_duration_seconds = len(window_pcm) / max(bytes_per_second, 1)
        window_end_ms = int(meta.get('window_end_ms', timestamp_ms))

        if not force and window_duration_seconds < self._min_window_seconds:
            return False

        last_emitted_at_ms = self._last_emitted_at_ms.get(stream_key)
        if force and last_emitted_at_ms is not None and window_end_ms <= last_emitted_at_ms:
            return False

        if (
            not force
            and last_emitted_at_ms is not None
            and (timestamp_ms - last_emitted_at_ms) < self._min_interval_ms
        ):
            return False

        enriched_meta = dict(meta)
        enriched_meta['window_duration_ms'] = int(window_duration_seconds * 1000)

        sr = int(meta.get('sample_rate', 0) or 0)
        ch = max(int(meta.get('channels', 1)), 1)
        wstats = compute_pcm_window_stats(
            window_pcm,
            sample_rate=sr,
            channels=ch,
        )
        samples_n = max(int(wstats.get('samples_count') or 0), 1)
        speech_ratio = float(wstats.get('speech_count') or 0) / float(samples_n)
        enriched_meta['speech_ratio'] = speech_ratio
        enriched_meta['mean_rms_dbfs'] = wstats.get('mean_rms_dbfs')
        logger.info(
            '🔊 Window ready | stream_key=%s | pcm_bytes=%s | duration_ms=%s | '
            'sample_rate=%s | channels=%s | window_start_ms=%s | window_end_ms=%s | '
            'seq=%s | mean_rms_dbfs=%s | speech_ratio=%.4f | peak_abs=%s',
            stream_key,
            len(window_pcm),
            enriched_meta['window_duration_ms'],
            sr,
            ch,
            int(meta.get('window_start_ms', 0)),
            int(meta.get('window_end_ms', 0)),
            int(meta.get('sequence', 0)),
            wstats.get('mean_rms_dbfs'),
            speech_ratio,
            wstats.get('peak_abs'),
        )

        self._window_callback(stream_key, window_pcm, enriched_meta)
        self._last_emitted_at_ms[stream_key] = window_end_ms
        return True

