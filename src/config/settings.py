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
    log_level: str = 'INFO'
    proto_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        # Railway injects `RAILWAY_SERVICE_NAME` in production containers; use it to
        # switch defaults from local dev to production endpoints.
        default_grpc_feedback_url = 'https://backend-analysis-production-a688.up.railway.app:50052'

        return cls(
            grpc_port=int(os.getenv('GRPC_PORT', '50051')),
            grpc_workers=int(os.getenv('GRPC_WORKERS', '10')),
            grpc_feedback_url=os.getenv('GRPC_FEEDBACK_URL', default_grpc_feedback_url),
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
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            proto_dir=os.getenv('PROTO_DIR'),
        )

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


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.validate()
    return _settings
