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
    # Backend service JWT (role=SERVICE) — required for multi-tenant ingress.
    # Read from BACKEND_SERVICE_TOKEN at startup; never log its value.
    grpc_feedback_service_token: Optional[str] = None
    # Optional: mint/refresh JWT via POST /auth/service-token (same SERVICE_BOOTSTRAP_KEY as backend).
    backend_http_base_url: Optional[str] = None
    service_bootstrap_key: Optional[str] = None
    service_token_mint_ttl_seconds: int = 3600
    service_token_mint_retries: int = 4
    service_token_mint_backoff_seconds: float = 1.0
    playbook_url_allowlist: str = ''
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
    # STT provider: `assemblyai` is the streaming cloud provider; `local` keeps
    # the legacy faster-whisper path available during rollout.
    stt_provider: str = 'local'
    # `multimodal` sends bounded PCM windows directly to Gemini. `transcript`
    # preserves AssemblyAI/local STT as a rollback path. `live` uses Gemini Live
    # for client audio (sub-second best-effort) with multimodal as degraded fallback.
    audio_analysis_mode: str = 'transcript'
    audio_analysis_client_interval_ms: int = 7000
    audio_analysis_host_interval_ms: int = 20000
    audio_analysis_overlap_ms: int = 750
    # Gemini Live guardrails / tuning
    live_model: str = 'gemini-3.1-flash-live-preview'
    live_silence_duration_ms: int = 250
    live_min_speech_ms: int = 400
    live_max_cost_usd_per_meeting: float = 3.0
    live_alert_cost_usd: float = 1.0
    live_max_concurrent_sessions: int = 20
    live_context_window_tokens: int = 12_000
    live_session_rotation_minutes: float = 2.0
    live_host_observe_interval_ms: int = 15_000
    live_langgraph_enabled: bool = True
    live_specialist_enabled: bool = False
    live_specialist_model: str = 'gemini-2.5-flash'
    live_specialist_queue_max_size: int = 32
    live_specialist_timeout_ms: int = 8_000
    live_specialist_cooldown_ms: int = 15_000
    live_specialist_min_confidence: float = 0.7
    live_specialist_max_age_ms: int = 120_000
    live_secondary_feedback_enabled: bool = True
    live_secondary_feedback_types: tuple[str, ...] = ('risk', 'objection')
    assemblyai_api_key: Optional[str] = None
    assemblyai_api_host: str = 'streaming.assemblyai.com'
    assemblyai_speech_model: str = 'u3-rt-pro'
    assemblyai_sample_rate: int = 16000
    assemblyai_format_turns: bool = True
    assemblyai_continuous_partials: bool = False
    assemblyai_stream_idle_timeout_ms: int = 30_000
    assemblyai_reconnect_limit: int = 2
    assemblyai_connect_timeout_seconds: float = 10.0
    assemblyai_termination_timeout_seconds: float = 2.0
    assemblyai_end_of_turn_confidence_threshold: Optional[float] = None
    assemblyai_min_turn_silence_ms: Optional[int] = None
    assemblyai_max_turn_silence_ms: Optional[int] = None
    assemblyai_vad_threshold: Optional[float] = None
    assemblyai_keyterms_prompt: Optional[str] = None
    assemblyai_tab_audio_vad_threshold: Optional[float] = None
    assemblyai_tab_audio_max_turn_silence_ms: Optional[int] = None
    partial_analysis_enabled: bool = True
    partial_stable_ms: int = 600
    partial_word_stable_ms: int = 300
    partial_growth_window_ms: int = 400
    partial_min_words: int = 5
    partial_cooldown_ms: int = 3000
    partial_min_confidence: float = 0.7
    feedback_allow_host_publish: bool = False
    # Seller Rooms acoustic routing: kill switch + shadow (classify, don't change routing).
    acoustic_routing_enabled: bool = True
    acoustic_shadow_mode: bool = False
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
    # Direct desktop WebSocket gateway (backend bypass on the critical path).
    # Binds to PORT (Railway public port, typically 8000) — same port the
    # platform routes wss://*.up.railway.app to. gRPC stays on GRPC_PORT.
    desktop_ws_enabled: bool = False
    port: int = 8765
    desktop_ws_coalesce_ms: int = 100
    desktop_ws_require_auth: bool = True
    # Same key material the backend uses (JWT_PUBLIC_KEY RS256 prod,
    # JWT_SECRET HS256 dev) so the gateway validates identical access tokens.
    jwt_public_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_issuer: str = 'meet-backend'
    jwt_audience: str = 'meet-platform'
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
    gemini_api_keys: tuple[str, ...] = ()
    gemini_model: str = 'gemini-2.5-flash'
    gemini_rpm_limit: int = 12
    gemini_rpm_window_sec: float = 60.0
    gemini_key_routing: str = 'tenant'

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
            grpc_feedback_service_token=(os.getenv('BACKEND_SERVICE_TOKEN') or None),
            backend_http_base_url=(
                (os.getenv('BACKEND_HTTP_BASE_URL') or '').strip() or None
            ),
            service_bootstrap_key=(
                (os.getenv('SERVICE_BOOTSTRAP_KEY') or '').strip() or None
            ),
            service_token_mint_ttl_seconds=max(
                60,
                min(
                    3600,
                    int(os.getenv('SERVICE_TOKEN_MINT_TTL_SECONDS', '3600')),
                ),
            ),
            service_token_mint_retries=max(
                1,
                int(os.getenv('SERVICE_TOKEN_MINT_RETRIES', '4')),
            ),
            service_token_mint_backoff_seconds=max(
                0.0,
                float(os.getenv('SERVICE_TOKEN_MINT_BACKOFF_SECONDS', '1.0')),
            ),
            playbook_url_allowlist=(
                os.getenv('PLAYBOOK_URL_ALLOWLIST') or ''
            ).strip(),
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
            stt_provider=os.getenv('STT_PROVIDER', 'assemblyai').strip().lower(),
            audio_analysis_mode=os.getenv(
                'AUDIO_ANALYSIS_MODE',
                'transcript',
            ).strip().lower(),
            audio_analysis_client_interval_ms=int(
                os.getenv('AUDIO_ANALYSIS_CLIENT_INTERVAL_MS', '7000'),
            ),
            audio_analysis_host_interval_ms=int(
                os.getenv('AUDIO_ANALYSIS_HOST_INTERVAL_MS', '20000'),
            ),
            audio_analysis_overlap_ms=int(
                os.getenv('AUDIO_ANALYSIS_OVERLAP_MS', '750'),
            ),
            live_model=os.getenv(
                'LIVE_MODEL',
                'gemini-3.1-flash-live-preview',
            ).strip()
            or 'gemini-3.1-flash-live-preview',
            live_silence_duration_ms=int(
                os.getenv('LIVE_SILENCE_DURATION_MS', '250'),
            ),
            live_min_speech_ms=int(
                os.getenv('LIVE_MIN_SPEECH_MS', '400'),
            ),
            live_max_cost_usd_per_meeting=float(
                os.getenv('LIVE_MAX_COST_USD_PER_MEETING', '3.0'),
            ),
            live_alert_cost_usd=float(
                os.getenv('LIVE_ALERT_COST_USD', '1.0'),
            ),
            live_max_concurrent_sessions=int(
                os.getenv('LIVE_MAX_CONCURRENT_SESSIONS', '20'),
            ),
            live_context_window_tokens=int(
                os.getenv('LIVE_CONTEXT_WINDOW_TOKENS', '12000'),
            ),
            live_session_rotation_minutes=float(
                os.getenv('LIVE_SESSION_ROTATION_MINUTES', '2.0'),
            ),
            live_host_observe_interval_ms=int(
                os.getenv('LIVE_HOST_OBSERVE_INTERVAL_MS', '15000'),
            ),
            live_langgraph_enabled=os.getenv(
                'LIVE_LANGGRAPH_ENABLED',
                'true',
            ).lower()
            == 'true',
            live_specialist_enabled=os.getenv(
                'LIVE_SPECIALIST_ENABLED',
                'false',
            ).lower()
            == 'true',
            live_specialist_model=os.getenv(
                'LIVE_SPECIALIST_MODEL',
                os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
            ).strip()
            or 'gemini-2.5-flash',
            live_specialist_queue_max_size=int(
                os.getenv('LIVE_SPECIALIST_QUEUE_MAX_SIZE', '32'),
            ),
            live_specialist_timeout_ms=int(
                os.getenv('LIVE_SPECIALIST_TIMEOUT_MS', '8000'),
            ),
            live_specialist_cooldown_ms=int(
                os.getenv('LIVE_SPECIALIST_COOLDOWN_MS', '15000'),
            ),
            live_specialist_min_confidence=float(
                os.getenv('LIVE_SPECIALIST_MIN_CONFIDENCE', '0.7'),
            ),
            live_specialist_max_age_ms=int(
                os.getenv('LIVE_SPECIALIST_MAX_AGE_MS', '120000'),
            ),
            live_secondary_feedback_enabled=os.getenv(
                'LIVE_SECONDARY_FEEDBACK_ENABLED',
                'true',
            ).lower()
            == 'true',
            live_secondary_feedback_types=cls._parse_csv(
                os.getenv('LIVE_SECONDARY_FEEDBACK_TYPES', 'risk,objection'),
            ),
            assemblyai_api_key=(os.getenv('ASSEMBLYAI_API_KEY') or '').strip() or None,
            assemblyai_api_host=os.getenv(
                'ASSEMBLYAI_API_HOST',
                'streaming.assemblyai.com',
            ).strip() or 'streaming.assemblyai.com',
            assemblyai_speech_model=os.getenv(
                'ASSEMBLYAI_SPEECH_MODEL',
                'u3-rt-pro',
            ).strip() or 'u3-rt-pro',
            assemblyai_sample_rate=int(os.getenv('ASSEMBLYAI_SAMPLE_RATE', '16000')),
            assemblyai_format_turns=os.getenv(
                'ASSEMBLYAI_FORMAT_TURNS',
                'true',
            ).lower()
            == 'true',
            assemblyai_continuous_partials=os.getenv(
                'ASSEMBLYAI_CONTINUOUS_PARTIALS',
                'false',
            ).lower()
            == 'true',
            assemblyai_stream_idle_timeout_ms=int(
                os.getenv('ASSEMBLYAI_STREAM_IDLE_TIMEOUT_MS', '30000'),
            ),
            assemblyai_reconnect_limit=int(
                os.getenv('ASSEMBLYAI_RECONNECT_LIMIT', '2'),
            ),
            assemblyai_connect_timeout_seconds=float(
                os.getenv('ASSEMBLYAI_CONNECT_TIMEOUT_SECONDS', '10.0'),
            ),
            assemblyai_termination_timeout_seconds=float(
                os.getenv('ASSEMBLYAI_TERMINATION_TIMEOUT_SECONDS', '2.0'),
            ),
            assemblyai_end_of_turn_confidence_threshold=cls._optional_float(
                os.getenv('ASSEMBLYAI_END_OF_TURN_CONFIDENCE_THRESHOLD'),
            ),
            assemblyai_min_turn_silence_ms=cls._optional_int(
                os.getenv('ASSEMBLYAI_MIN_TURN_SILENCE_MS'),
            ),
            assemblyai_max_turn_silence_ms=cls._optional_int(
                os.getenv('ASSEMBLYAI_MAX_TURN_SILENCE_MS'),
            ),
            assemblyai_vad_threshold=cls._optional_float(
                os.getenv('ASSEMBLYAI_VAD_THRESHOLD'),
            ),
            assemblyai_keyterms_prompt=(
                (os.getenv('ASSEMBLYAI_KEYTERMS_PROMPT') or '').strip() or None
            ),
            assemblyai_tab_audio_vad_threshold=cls._optional_float(
                os.getenv('ASSEMBLYAI_TAB_AUDIO_VAD_THRESHOLD'),
            ),
            assemblyai_tab_audio_max_turn_silence_ms=cls._optional_int(
                os.getenv('ASSEMBLYAI_TAB_AUDIO_MAX_TURN_SILENCE_MS'),
            ),
            partial_analysis_enabled=os.getenv(
                'PARTIAL_ANALYSIS_ENABLED',
                'true',
            ).lower()
            == 'true',
            partial_stable_ms=int(os.getenv('PARTIAL_STABLE_MS', '600')),
            partial_word_stable_ms=int(os.getenv('PARTIAL_WORD_STABLE_MS', '300')),
            partial_growth_window_ms=int(
                os.getenv('PARTIAL_GROWTH_WINDOW_MS', '400'),
            ),
            partial_min_words=int(os.getenv('PARTIAL_MIN_WORDS', '5')),
            partial_cooldown_ms=int(os.getenv('PARTIAL_COOLDOWN_MS', '3000')),
            partial_min_confidence=float(os.getenv('PARTIAL_MIN_CONFIDENCE', '0.7')),
            feedback_allow_host_publish=os.getenv(
                'FEEDBACK_ALLOW_HOST_PUBLISH',
                'false',
            ).lower()
            == 'true',
            acoustic_routing_enabled=os.getenv(
                'ACOUSTIC_ROUTING_ENABLED',
                'true',
            ).lower()
            == 'true',
            acoustic_shadow_mode=os.getenv(
                'ACOUSTIC_SHADOW_MODE',
                'false',
            ).lower()
            == 'true',
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
            desktop_ws_enabled=os.getenv('DESKTOP_WS_ENABLED', 'false').lower()
            == 'true',
            port=int(os.getenv('PORT', '8765')),
            desktop_ws_coalesce_ms=int(os.getenv('DESKTOP_WS_COALESCE_MS', '100')),
            desktop_ws_require_auth=os.getenv(
                'DESKTOP_WS_REQUIRE_AUTH',
                'true',
            ).lower()
            == 'true',
            jwt_public_key=(os.getenv('JWT_PUBLIC_KEY') or '').strip() or None,
            jwt_secret=(os.getenv('JWT_SECRET') or '').strip() or None,
            jwt_issuer=os.getenv('JWT_ISSUER', 'meet-backend').strip()
            or 'meet-backend',
            jwt_audience=os.getenv('JWT_AUDIENCE', 'meet-platform').strip()
            or 'meet-platform',
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
            gemini_api_key=(
                (os.getenv('GEMINI_API_KEY') or '').strip().strip('"').strip("'")
                or None
            ),
            gemini_api_keys=cls._parse_csv(os.getenv('GEMINI_API_KEYS')),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
            gemini_rpm_limit=int(os.getenv('GEMINI_RPM_LIMIT', '12')),
            gemini_rpm_window_sec=float(os.getenv('GEMINI_RPM_WINDOW_SEC', '60.0')),
            gemini_key_routing=os.getenv('GEMINI_KEY_ROUTING', 'tenant').strip().lower(),
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

    @staticmethod
    def _optional_int(raw: Optional[str]) -> Optional[int]:
        if raw is None or not raw.strip():
            return None
        return int(raw)

    @staticmethod
    def _optional_float(raw: Optional[str]) -> Optional[float]:
        if raw is None or not raw.strip():
            return None
        return float(raw)

    @staticmethod
    def _parse_csv(raw: Optional[str]) -> tuple[str, ...]:
        if raw is None or not raw.strip():
            return ()
        cleaned = raw.strip()
        if (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in {'"', "'"}
        ):
            cleaned = cleaned[1:-1]
        parts = [part.strip().strip('"').strip("'") for part in cleaned.split(',')]
        if any(not part for part in parts):
            raise ValueError('Gemini API key list contains empty entries.')
        return tuple(parts)

    def effective_gemini_api_keys(self) -> tuple[str, ...]:
        """Return all configured Gemini API keys for the key pool.

        ``GEMINI_API_KEYS`` takes precedence. If only ``GEMINI_API_KEY`` is set and it
        contains commas, split it the same way (common misconfiguration).
        """
        if self.gemini_api_keys:
            return self.gemini_api_keys
        single = (self.gemini_api_key or '').strip()
        if not single:
            return ()
        if ',' in single:
            return self._parse_csv(single)
        return (single,)

    def grpc_feedback_wants_auto_jwt(self) -> bool:
        """True when env requests automatic SERVICE JWT mint/refresh via HTTP bootstrap.

        Static BACKEND_SERVICE_TOKEN has precedence. If a static token is present,
        we avoid auto-mint so transient backend HTTP issues do not block publishes.
        """
        has_static = bool((self.grpc_feedback_service_token or '').strip())
        return bool(
            not has_static
            and
            self.backend_http_base_url
            and self.service_bootstrap_key,
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
        if self.grpc_feedback_enabled:
            has_static = bool((self.grpc_feedback_service_token or '').strip())
            wants_mint = self.grpc_feedback_wants_auto_jwt()
            mint_partial = (
                bool((self.service_bootstrap_key or '').strip())
                or bool((self.backend_http_base_url or '').strip())
            ) and not wants_mint and not has_static
            if mint_partial:
                raise ValueError(
                    'Incomplete automatic service JWT config: set all of '
                    'SERVICE_BOOTSTRAP_KEY and '
                    'BACKEND_HTTP_BASE_URL (or remove partial values).',
                )
            if not has_static and not wants_mint:
                raise ValueError(
                    'GRPC_FEEDBACK_ENABLED is true but no backend auth is configured. '
                    'Set BACKEND_SERVICE_TOKEN, or set SERVICE_BOOTSTRAP_KEY + '
                    'BACKEND_HTTP_BASE_URL for '
                    'automatic renewal.',
                )
            if self.service_token_mint_retries < 1:
                raise ValueError(
                    f'Invalid SERVICE_TOKEN_MINT_RETRIES: {self.service_token_mint_retries}',
                )
            if self.service_token_mint_backoff_seconds < 0:
                raise ValueError(
                    'Invalid SERVICE_TOKEN_MINT_BACKOFF_SECONDS: '
                    f'{self.service_token_mint_backoff_seconds}',
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
        if self.stt_provider not in {'assemblyai', 'local'}:
            raise ValueError(
                f'Invalid STT_PROVIDER: {self.stt_provider}. '
                'Expected "assemblyai" or "local".',
            )
        if self.audio_analysis_mode not in {'transcript', 'multimodal', 'live'}:
            raise ValueError(
                f'Invalid AUDIO_ANALYSIS_MODE: {self.audio_analysis_mode}. '
                'Expected "transcript", "multimodal", or "live".',
            )
        if self.audio_analysis_client_interval_ms < 1000:
            raise ValueError('AUDIO_ANALYSIS_CLIENT_INTERVAL_MS must be >= 1000.')
        if self.audio_analysis_host_interval_ms < 1000:
            raise ValueError('AUDIO_ANALYSIS_HOST_INTERVAL_MS must be >= 1000.')
        if self.audio_analysis_overlap_ms < 0:
            raise ValueError('AUDIO_ANALYSIS_OVERLAP_MS must be >= 0.')
        if self.audio_analysis_mode in {'multimodal', 'live'} and self.llm_provider != 'gemini':
            raise ValueError(
                f'AUDIO_ANALYSIS_MODE={self.audio_analysis_mode} requires LLM_PROVIDER=gemini.',
            )
        if self.audio_analysis_mode == 'live':
            if self.live_silence_duration_ms < 50:
                raise ValueError('LIVE_SILENCE_DURATION_MS must be >= 50.')
            if self.live_min_speech_ms < 0:
                raise ValueError('LIVE_MIN_SPEECH_MS must be >= 0.')
            if self.live_max_cost_usd_per_meeting <= 0:
                raise ValueError('LIVE_MAX_COST_USD_PER_MEETING must be > 0.')
            if self.live_alert_cost_usd < 0:
                raise ValueError('LIVE_ALERT_COST_USD must be >= 0.')
            if self.live_max_concurrent_sessions < 1:
                raise ValueError('LIVE_MAX_CONCURRENT_SESSIONS must be >= 1.')
            if self.live_context_window_tokens < 1000:
                raise ValueError('LIVE_CONTEXT_WINDOW_TOKENS must be >= 1000.')
            if self.live_session_rotation_minutes <= 0:
                raise ValueError('LIVE_SESSION_ROTATION_MINUTES must be > 0.')
            if self.live_host_observe_interval_ms < 0:
                raise ValueError('LIVE_HOST_OBSERVE_INTERVAL_MS must be >= 0.')
            if self.live_specialist_queue_max_size < 1:
                raise ValueError('LIVE_SPECIALIST_QUEUE_MAX_SIZE must be >= 1.')
            if self.live_specialist_timeout_ms < 100:
                raise ValueError('LIVE_SPECIALIST_TIMEOUT_MS must be >= 100.')
            if self.live_specialist_cooldown_ms < 0:
                raise ValueError('LIVE_SPECIALIST_COOLDOWN_MS must be >= 0.')
            if not 0.0 <= self.live_specialist_min_confidence <= 1.0:
                raise ValueError(
                    'LIVE_SPECIALIST_MIN_CONFIDENCE must be between 0 and 1.',
                )
            if self.live_specialist_max_age_ms < 100:
                raise ValueError('LIVE_SPECIALIST_MAX_AGE_MS must be >= 100.')
            invalid_secondary_types = set(self.live_secondary_feedback_types) - {
                'risk',
                'objection',
            }
            if invalid_secondary_types:
                raise ValueError(
                    'LIVE_SECONDARY_FEEDBACK_TYPES supports only risk,objection.',
                )
        if self.audio_analysis_mode == 'transcript' and self.stt_provider == 'assemblyai':
            if not (self.assemblyai_api_key or '').strip():
                raise ValueError(
                    'STT_PROVIDER=assemblyai requires ASSEMBLYAI_API_KEY.',
                )
            if self.assemblyai_sample_rate < 8000:
                raise ValueError(
                    f'Invalid ASSEMBLYAI_SAMPLE_RATE: {self.assemblyai_sample_rate}',
                )
            if self.assemblyai_stream_idle_timeout_ms < 1000:
                raise ValueError(
                    'Invalid ASSEMBLYAI_STREAM_IDLE_TIMEOUT_MS: '
                    f'{self.assemblyai_stream_idle_timeout_ms}',
                )
            if self.assemblyai_reconnect_limit < 0:
                raise ValueError(
                    f'Invalid ASSEMBLYAI_RECONNECT_LIMIT: {self.assemblyai_reconnect_limit}',
                )
            if self.assemblyai_connect_timeout_seconds <= 0:
                raise ValueError(
                    'Invalid ASSEMBLYAI_CONNECT_TIMEOUT_SECONDS: '
                    f'{self.assemblyai_connect_timeout_seconds}',
                )
            if self.assemblyai_termination_timeout_seconds < 0:
                raise ValueError(
                    'Invalid ASSEMBLYAI_TERMINATION_TIMEOUT_SECONDS: '
                    f'{self.assemblyai_termination_timeout_seconds}',
                )
            if (
                self.assemblyai_end_of_turn_confidence_threshold is not None
                and not 0.0 <= self.assemblyai_end_of_turn_confidence_threshold <= 1.0
            ):
                raise ValueError(
                    'Invalid ASSEMBLYAI_END_OF_TURN_CONFIDENCE_THRESHOLD: '
                    f'{self.assemblyai_end_of_turn_confidence_threshold}',
                )
            if (
                self.assemblyai_vad_threshold is not None
                and not 0.0 <= self.assemblyai_vad_threshold <= 1.0
            ):
                raise ValueError(
                    f'Invalid ASSEMBLYAI_VAD_THRESHOLD: {self.assemblyai_vad_threshold}',
                )
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
        if self.desktop_ws_enabled:
            if self.port < 1 or self.port > 65535:
                raise ValueError(f'Invalid PORT: {self.port}')
            if self.desktop_ws_coalesce_ms < 20:
                raise ValueError(
                    f'Invalid DESKTOP_WS_COALESCE_MS: {self.desktop_ws_coalesce_ms} '
                    '(min 20ms)',
                )
            if self.desktop_ws_require_auth and not (
                self.jwt_public_key or self.jwt_secret
            ):
                raise ValueError(
                    'DESKTOP_WS_ENABLED=true requires JWT_PUBLIC_KEY (RS256) or '
                    'JWT_SECRET (HS256, dev only) to validate desktop tokens. '
                    'Set DESKTOP_WS_REQUIRE_AUTH=false only on trusted networks.',
                )
        if self.llm_provider == 'gemini':
            keys = self.effective_gemini_api_keys()
            if not keys:
                raise ValueError(
                    'LLM_PROVIDER=gemini requires GEMINI_API_KEYS or GEMINI_API_KEY.',
                )
            if self.gemini_rpm_limit < 1:
                raise ValueError(f'Invalid GEMINI_RPM_LIMIT: {self.gemini_rpm_limit}')
            if self.gemini_rpm_window_sec <= 0:
                raise ValueError(
                    f'Invalid GEMINI_RPM_WINDOW_SEC: {self.gemini_rpm_window_sec}',
                )
            if self.gemini_key_routing not in {'tenant'}:
                raise ValueError(
                    f'Invalid GEMINI_KEY_ROUTING: {self.gemini_key_routing}. '
                    'Expected "tenant".',
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
