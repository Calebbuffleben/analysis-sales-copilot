"""TranscriptionService: low-level service that turns audio windows into text.

This module is responsible only for calling the STT provider. It does not decide
what happens to the transcript; that is the responsibility of
TranscriptionPipelineService.

STT provider: faster-whisper (>= 1.0.0), which depends on ctranslate2.

Concurrency:
- **In-process (default, STT_PROCESS_WORKERS=0)**: one WhisperModel guarded by a
  threading lock (faster-whisper is not safe for concurrent transcribe calls).
- **Process pool (STT_PROCESS_WORKERS>=1)**: one WhisperModel per worker process;
  parallel transcribes scale with worker count (Phase 5).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Optional

from .transcription_core import TranscriptionSttConfig, transcribe_pcm_window
from .types import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Turns a PCM window into transcript text using faster-whisper."""

    def __init__(
        self,
        model_size: str = 'small',
        device: str = 'cpu',
        compute_type: str = 'int8',
        vad_filter: bool = True,
        empty_diagnostic_no_vad: bool = False,
        low_energy_dbfs_threshold: float = -50.0,
        default_language: Optional[str] = None,
        process_workers: int = 0,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._vad_filter = vad_filter
        self._empty_diagnostic_no_vad = empty_diagnostic_no_vad
        self._low_energy_dbfs_threshold = low_energy_dbfs_threshold
        self._default_language = self._normalize_language(default_language)
        self._process_workers = max(0, int(process_workers))

        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._diagnostic_executor: Optional[ThreadPoolExecutor] = None
        if empty_diagnostic_no_vad:
            self._diagnostic_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='stt-no-vad-diag',
            )

        self._stt_config = TranscriptionSttConfig(
            vad_filter=self._vad_filter,
            low_energy_dbfs_threshold=self._low_energy_dbfs_threshold,
            default_language=self._default_language,
            empty_diagnostic_no_vad=self._empty_diagnostic_no_vad,
        )

        self._process_pool: Optional[ProcessPoolExecutor] = None
        if self._process_workers > 0:
            from .transcription_process_pool import (
                pool_init_worker,
                pool_transcribe_job,
            )

            self._process_pool = ProcessPoolExecutor(
                max_workers=self._process_workers,
                initializer=pool_init_worker,
                initargs=(
                    self._model_size,
                    self._device,
                    self._compute_type,
                    self._vad_filter,
                    self._low_energy_dbfs_threshold,
                    self._default_language,
                    self._empty_diagnostic_no_vad,
                ),
            )
            self._pool_transcribe_job = pool_transcribe_job
            logger.info(
                'STT process pool enabled | process_workers=%s (one Whisper per process)',
                self._process_workers,
            )

    def transcribe(self, window_pcm: bytes, meta: dict) -> TranscriptionResult:
        """Transcribe a single PCM audio window."""
        if self._process_pool is not None:
            future = self._process_pool.submit(
                self._pool_transcribe_job,
                (window_pcm, meta),
            )
            return future.result()

        with self._model_lock:
            return transcribe_pcm_window(
                self._get_model(),
                window_pcm,
                meta,
                config=self._stt_config,
                diagnostic_executor=self._diagnostic_executor,
                model_lock=self._model_lock,
            )

    def _get_model(self) -> Any:
        """Lazily create and cache the faster-whisper model instance."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    'faster-whisper is not installed. '
                    'Install project requirements before running transcription.',
                ) from error

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

        return self._model

    def preload_model(self) -> None:
        """Load Whisper into memory before the first real window."""
        if self._process_pool is not None:
            from .transcription_process_pool import pool_warmup

            # Force each worker process to start so initializers run (model load).
            futures = [
                self._process_pool.submit(pool_warmup, i)
                for i in range(self._process_workers)
            ]
            for f in futures:
                f.result()
            logger.info(
                'Whisper process pool warmed | process_workers=%s | model_size=%s',
                self._process_workers,
                self._model_size,
            )
            return

        with self._model_lock:
            self._get_model()
        logger.info(
            'Whisper model ready | model_size=%s | device=%s | compute_type=%s',
            self._model_size,
            self._device,
            self._compute_type,
        )

    def shutdown(self) -> None:
        """Release process pool (best-effort on server stop)."""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=False)
            self._process_pool = None
        if self._diagnostic_executor is not None:
            self._diagnostic_executor.shutdown(wait=False)
            self._diagnostic_executor = None

    @staticmethod
    def _normalize_language(language: Optional[object]) -> Optional[str]:
        if language is None:
            return None
        value = str(language).strip().lower()
        return value or None
