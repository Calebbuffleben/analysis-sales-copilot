"""Application settings and configuration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application configuration settings."""

    grpc_port: int = 50051
    grpc_workers: int = 10
    grpc_feedback_url: str = 'localhost:50052'
    grpc_feedback_enabled: bool = True
    grpc_feedback_timeout_seconds: float = 5.0
    storage_dir: str = '/app/storage'
    audio_buffer_window_seconds: float = 10.0
    audio_buffer_min_window_seconds: float = 4.0
    audio_buffer_min_interval_ms: int = 2000
    transcription_model_size: str = 'small'
    transcription_device: str = 'cpu'
    transcription_compute_type: str = 'int8'
    whisper_vad_filter: bool = True
    whisper_empty_diagnostic_no_vad: bool = False
    whisper_low_energy_dbfs: float = -50.0
    whisper_default_language: Optional[str] = None
    # Ready-window queue (bounded realtime processing)
    window_queue_max_size: int = 8
    window_worker_threads: int = 2
    window_max_age_ms: int = 25_000
    window_low_priority_speech_ratio_below: float = 0.02
    log_level: str = 'INFO'
    proto_dir: Optional[str] = None
    # Load Whisper + sentence-transformers before accepting traffic (avoids multi-minute
    # delay on first real-time window from HF download + model init).
    preload_ml_models: bool = True

    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        # Backend gRPC ingress is plain (insecure) on 50052. On Railway, reach it via
        # private DNS: <backend-service>.railway.internal:50052 — not https://*.up.railway.app
        default_grpc_feedback_url = (
            'backend-analysis-production.railway.internal:50052'
            if os.getenv('RAILWAY_SERVICE_NAME')
            else 'localhost:50052'
        )
        feedback_raw = os.getenv('GRPC_FEEDBACK_URL', default_grpc_feedback_url)
        grpc_feedback_url = cls._normalize_grpc_target(feedback_raw)

        return cls(
            grpc_port=int(os.getenv('GRPC_PORT', '50051')),
            grpc_workers=int(os.getenv('GRPC_WORKERS', '10')),
            grpc_feedback_url=grpc_feedback_url,
            grpc_feedback_enabled=os.getenv('GRPC_FEEDBACK_ENABLED', 'true').lower() == 'true',
            grpc_feedback_timeout_seconds=float(
                os.getenv('GRPC_FEEDBACK_TIMEOUT_SECONDS', '5.0'),
            ),
            storage_dir=os.getenv('STORAGE_DIR', '/app/storage'),
            audio_buffer_window_seconds=float(
                os.getenv('AUDIO_BUFFER_WINDOW_SECONDS', '10.0'),
            ),
            audio_buffer_min_window_seconds=float(
                os.getenv('AUDIO_BUFFER_MIN_WINDOW_SECONDS', '4.0'),
            ),
            audio_buffer_min_interval_ms=int(
                os.getenv('AUDIO_BUFFER_MIN_INTERVAL_MS', '2000'),
            ),
            transcription_model_size=os.getenv('TRANSCRIPTION_MODEL_SIZE', 'small'),
            transcription_device=os.getenv('TRANSCRIPTION_DEVICE', 'cpu'),
            transcription_compute_type=os.getenv(
                'TRANSCRIPTION_COMPUTE_TYPE',
                'int8',
            ),
            whisper_vad_filter=os.getenv('WHISPER_VAD_FILTER', 'true').lower()
            == 'true',
            whisper_empty_diagnostic_no_vad=os.getenv(
                'WHISPER_EMPTY_DIAGNOSTIC_NO_VAD',
                'false',
            ).lower()
            == 'true',
            whisper_low_energy_dbfs=float(
                os.getenv('WHISPER_LOW_ENERGY_DBFS', '-50.0'),
            ),
            whisper_default_language=cls._normalize_language(
                os.getenv('WHISPER_DEFAULT_LANGUAGE'),
            ),
            window_queue_max_size=int(os.getenv('WINDOW_QUEUE_MAX_SIZE', '8')),
            window_worker_threads=int(os.getenv('WINDOW_WORKER_THREADS', '2')),
            window_max_age_ms=int(os.getenv('WINDOW_MAX_AGE_MS', '25000')),
            window_low_priority_speech_ratio_below=float(
                os.getenv('WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW', '0.02'),
            ),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            proto_dir=os.getenv('PROTO_DIR'),
            preload_ml_models=os.getenv('PRELOAD_ML_MODELS', 'true').lower() == 'true',
        )

    @staticmethod
    def _normalize_grpc_target(raw: str) -> str:
        """Strip http(s):// for grpc.insecure_channel (host:port only)."""
        if not raw:
            return raw
        u = raw.strip()
        if u.startswith('https://'):
            return u[8:].split('/', 1)[0]
        if u.startswith('http://'):
            return u[7:].split('/', 1)[0]
        return u

    @staticmethod
    def _normalize_language(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        value = raw.strip().lower()
        return value or None

    def validate(self) -> None:
        """Validate settings values."""
        if self.grpc_port < 1 or self.grpc_port > 65535:
            raise ValueError(f'Invalid GRPC_PORT: {self.grpc_port}')
        if self.grpc_workers < 1:
            raise ValueError(f'Invalid GRPC_WORKERS: {self.grpc_workers}')
        if self.grpc_feedback_timeout_seconds <= 0:
            raise ValueError(
                f'Invalid GRPC_FEEDBACK_TIMEOUT_SECONDS: {self.grpc_feedback_timeout_seconds}',
            )
        if self.audio_buffer_window_seconds <= 0:
            raise ValueError(
                f'Invalid AUDIO_BUFFER_WINDOW_SECONDS: {self.audio_buffer_window_seconds}',
            )
        if self.audio_buffer_min_window_seconds <= 0:
            raise ValueError(
                'Invalid AUDIO_BUFFER_MIN_WINDOW_SECONDS: '
                f'{self.audio_buffer_min_window_seconds}',
            )
        if self.audio_buffer_min_interval_ms < 0:
            raise ValueError(
                f'Invalid AUDIO_BUFFER_MIN_INTERVAL_MS: {self.audio_buffer_min_interval_ms}',
            )
        if not self.log_level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f'Invalid LOG_LEVEL: {self.log_level}')
        if not -120.0 <= self.whisper_low_energy_dbfs <= 0.0:
            raise ValueError(
                f'Invalid WHISPER_LOW_ENERGY_DBFS: {self.whisper_low_energy_dbfs}',
            )
        if self.window_queue_max_size < 1:
            raise ValueError(
                f'Invalid WINDOW_QUEUE_MAX_SIZE: {self.window_queue_max_size}',
            )
        if self.window_worker_threads < 1:
            raise ValueError(
                f'Invalid WINDOW_WORKER_THREADS: {self.window_worker_threads}',
            )
        if self.window_max_age_ms < 1000:
            raise ValueError(
                f'Invalid WINDOW_MAX_AGE_MS: {self.window_max_age_ms}',
            )
        if not 0.0 <= self.window_low_priority_speech_ratio_below <= 1.0:
            raise ValueError(
                'Invalid WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW: '
                f'{self.window_low_priority_speech_ratio_below}',
            )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.validate()
    return _settings
