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
    # STT process parallelism (Phase 5): 0 = in-process + lock; N>=1 = N worker processes,
    # each with its own WhisperModel (true parallel transcribe).
    stt_process_workers: int = 0
    # Ready-window queue (bounded realtime processing)
    window_queue_max_size: int = 8
    window_worker_threads: int = 2
    window_max_age_ms: int = 25_000
    window_low_priority_speech_ratio_below: float = 0.02
    # Publish dispatcher (decouple STT/analysis from backend gRPC I/O)
    publish_queue_max_size: int = 64
    publish_worker_threads: int = 2
    # Max wall time after window_end_ms before dropping publish. Must exceed worst-case
    # (queue + STT + analysis); 10s was too tight on CPU and dropped all gRPC publishes.
    publish_max_age_ms: int = 60_000
    publish_retry_limit: int = 1
    publish_retry_backoff_ms: int = 200
    metrics_enabled: bool = True
    metrics_port: int = 9100
    log_level: str = 'INFO'
    proto_dir: Optional[str] = None
    # Load Whisper + sentence-transformers before accepting traffic (avoids multi-minute
    # delay on first real-time window from HF download + model init).
    preload_ml_models: bool = True
    
    # ===========================================
    # LLM Configuration
    # ===========================================
    # LLM Provider: 'ollama' (free, local) or 'gemini' (Google API)
    llm_provider: str = 'ollama'
    
    # Ollama settings (for local free inference)
    ollama_base_url: str = 'http://localhost:11434'
    ollama_model: str = 'llama3.1:8b'
    ollama_timeout: int = 30
    
    # Gemini settings (if using Google API)
    gemini_api_key: Optional[str] = None

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
            stt_process_workers=int(os.getenv('STT_PROCESS_WORKERS', '0')),
            window_queue_max_size=int(os.getenv('WINDOW_QUEUE_MAX_SIZE', '8')),
            window_worker_threads=int(os.getenv('WINDOW_WORKER_THREADS', '2')),
            window_max_age_ms=int(os.getenv('WINDOW_MAX_AGE_MS', '25000')),
            window_low_priority_speech_ratio_below=float(
                os.getenv('WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW', '0.02'),
            ),
            publish_queue_max_size=int(
                os.getenv('PUBLISH_QUEUE_MAX_SIZE', '64'),
            ),
            publish_worker_threads=int(
                os.getenv('PUBLISH_WORKER_THREADS', '2'),
            ),
            publish_max_age_ms=int(os.getenv('PUBLISH_MAX_AGE_MS', '60000')),
            publish_retry_limit=int(os.getenv('PUBLISH_RETRY_LIMIT', '1')),
            publish_retry_backoff_ms=int(
                os.getenv('PUBLISH_RETRY_BACKOFF_MS', '200'),
            ),
            metrics_enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
            metrics_port=int(os.getenv('METRICS_PORT', '9100')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            proto_dir=os.getenv('PROTO_DIR'),
            preload_ml_models=os.getenv('PRELOAD_ML_MODELS', 'true').lower() == 'true',
            # LLM Provider settings
            llm_provider=os.getenv('LLM_PROVIDER', 'ollama').lower(),
            ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            ollama_model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
            ollama_timeout=int(os.getenv('OLLAMA_TIMEOUT', '30')),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
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
        if self.stt_process_workers < 0:
            raise ValueError(
                f'Invalid STT_PROCESS_WORKERS: {self.stt_process_workers}',
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
        if self.publish_queue_max_size < 1:
            raise ValueError(
                f'Invalid PUBLISH_QUEUE_MAX_SIZE: {self.publish_queue_max_size}',
            )
        if self.publish_worker_threads < 1:
            raise ValueError(
                f'Invalid PUBLISH_WORKER_THREADS: {self.publish_worker_threads}',
            )
        if self.publish_max_age_ms < 100:
            raise ValueError(
                f'Invalid PUBLISH_MAX_AGE_MS: {self.publish_max_age_ms}',
            )
        if self.publish_retry_limit < 0:
            raise ValueError(
                f'Invalid PUBLISH_RETRY_LIMIT: {self.publish_retry_limit}',
            )
        if self.publish_retry_backoff_ms < 0:
            raise ValueError(
                f'Invalid PUBLISH_RETRY_BACKOFF_MS: {self.publish_retry_backoff_ms}',
            )
        if self.metrics_port < 1 or self.metrics_port > 65535:
            raise ValueError(f'Invalid METRICS_PORT: {self.metrics_port}')


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.validate()
    return _settings
