"""Process-pool STT workers: one faster-whisper model per worker process."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple

from .transcription_core import TranscriptionSttConfig, transcribe_pcm_window
from .types import TranscriptionResult

logger = logging.getLogger(__name__)

_worker_model: Any = None
_worker_lock: Optional[threading.Lock] = None
_worker_diagnostic_executor: Optional[ThreadPoolExecutor] = None
_worker_config: Optional[TranscriptionSttConfig] = None


def pool_init_worker(
    model_size: str,
    device: str,
    compute_type: str,
    vad_filter: bool,
    low_energy_dbfs_threshold: float,
    default_language: Optional[str],
    empty_diagnostic_no_vad: bool,
) -> None:
    """Load one Whisper model per process (initializer for ProcessPoolExecutor)."""
    global _worker_model, _worker_lock, _worker_diagnostic_executor, _worker_config

    _worker_lock = threading.Lock()
    _worker_config = TranscriptionSttConfig(
        vad_filter=vad_filter,
        low_energy_dbfs_threshold=low_energy_dbfs_threshold,
        default_language=default_language,
        empty_diagnostic_no_vad=empty_diagnostic_no_vad,
    )
    if empty_diagnostic_no_vad:
        _worker_diagnostic_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='stt-no-vad-diag',
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            'faster-whisper is not installed. '
            'Install project requirements before running transcription.',
        ) from error

    _worker_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    logger.info(
        'STT process worker ready | model_size=%s | device=%s | compute_type=%s',
        model_size,
        device,
        compute_type,
    )


def pool_transcribe_job(payload: Tuple[bytes, dict]) -> TranscriptionResult:
    """Picklable entrypoint: transcribe one window in this worker process."""
    window_pcm, meta = payload
    assert _worker_model is not None and _worker_lock is not None and _worker_config is not None
    with _worker_lock:
        return transcribe_pcm_window(
            _worker_model,
            window_pcm,
            meta,
            config=_worker_config,
            diagnostic_executor=_worker_diagnostic_executor,
            model_lock=_worker_lock,
        )


def pool_warmup(_: int) -> bool:
    """Cheap job to force worker processes to start (initializer runs first)."""
    return _worker_model is not None
