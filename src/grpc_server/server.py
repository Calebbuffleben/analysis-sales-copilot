"""gRPC server setup and initialization."""

import logging
import os
import signal
import sys
import time
from concurrent import futures
from typing import Optional

import grpc

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'proto'))

import audio_pipeline_pb2_grpc

from ..config.settings import Settings, get_settings
from ..handlers.audio_handler import AudioPipelineServicer
from ..modules.audio_buffer.service import AudioBufferService
from ..modules.audio_buffer.sliding_worker import SlidingWindowWorker
from ..modules.backend_feedback.grpc_feedback_client import BackendFeedbackClient
from ..modules.backend_feedback.publish_dispatcher import PublishDispatcher
from ..modules.backend_feedback.service_jwt_provider import ServiceJwtProvider
from ..modules.text_analysis.text_analysis_service import TextAnalysisService
from ..feedback_trace import make_feedback_trace_id
from ..modules.transcription.assemblyai_streaming_provider import (
    AssemblyAiStreamConfig,
    AssemblyAiStreamingProvider,
)
from ..modules.transcription.partial_turn_coordinator import (
    PartialTurnConfig,
    PartialTurnCoordinator,
)
from ..modules.transcription.ready_window_dispatcher import ReadyWindowDispatcher
from ..pipeline_latency import LatencyTraceContext, log_assemblyai_partial_stable
from ..modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)
from ..modules.transcription.transcription_service import TranscriptionService
from ..services.audio_service import AudioService
from ..utils.proto_utils import (
    generate_proto_code_batch,
    validate_proto_file_list,
)

logger = logging.getLogger(__name__)


class _ServerRuntime:
    """Holds runtime-managed resources for graceful shutdown."""

    def __init__(
        self,
        text_analysis_service: TextAnalysisService,
        publish_dispatcher: PublishDispatcher,
        backend_feedback_client: BackendFeedbackClient,
        transcription_service: Optional[TranscriptionService] = None,
        streaming_stt_provider: Optional[AssemblyAiStreamingProvider] = None,
    ) -> None:
        self.text_analysis_service = text_analysis_service
        self.publish_dispatcher = publish_dispatcher
        self.backend_feedback_client = backend_feedback_client
        self.transcription_service = transcription_service
        self.streaming_stt_provider = streaming_stt_provider
        self._closed = False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self.text_analysis_service.shutdown()
        except Exception:
            logger.exception('Failed to shutdown text analysis service')

        try:
            self.publish_dispatcher.shutdown(wait=True)
        except Exception:
            logger.exception('Failed to shutdown publish dispatcher')

        try:
            self.backend_feedback_client.close()
        except Exception:
            logger.exception('Failed to close backend feedback client')

        try:
            if self.streaming_stt_provider is not None:
                self.streaming_stt_provider.close_all()
        except Exception:
            logger.exception('Failed to close streaming STT provider')

        try:
            if self.transcription_service is not None:
                self.transcription_service.shutdown()
        except Exception:
            logger.exception('Failed to shutdown transcription service')


def validate_proto_code(proto_dir: Optional[str] = None) -> bool:
    """
    Validate that proto code has been generated.

    Args:
        proto_dir: Directory containing proto files. If None, uses default location.

    Returns:
        True if proto code exists, False otherwise
    """
    if proto_dir is None:
        proto_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')

    proto_dir = os.path.abspath(proto_dir)
    required_proto_files = ['audio_pipeline.proto', 'feedback_ingestion.proto']
    required_generated_files = [
        ('audio_pipeline_pb2.py', 'audio_pipeline_pb2_grpc.py'),
        ('feedback_ingestion_pb2.py', 'feedback_ingestion_pb2_grpc.py'),
    ]

    missing_generated_files = [
        file_name
        for pair in required_generated_files
        for file_name in pair
        if not os.path.exists(os.path.join(proto_dir, file_name))
    ]

    if missing_generated_files:
        logger.warning("Código gRPC não encontrado. Tentando gerar...")

        # Validate proto files exist
        is_valid, error_msg = validate_proto_file_list(proto_dir, required_proto_files)
        if not is_valid:
            logger.error(f"Validação de arquivo proto falhou: {error_msg}")
            return False

        # Generate proto code
        if not generate_proto_code_batch(
            proto_dir=proto_dir,
            proto_files=required_proto_files,
        ):
            logger.error("Falha ao gerar código gRPC")
            return False

    return True


def _warmup_ml_models(
    transcription_service: TranscriptionService,
    text_analysis_service: TextAnalysisService,
) -> None:
    """Load Whisper before the first audio window (avoids cold-start lag)."""
    t0 = time.perf_counter()
    try:
        transcription_service.preload_model()
    except Exception:
        logger.exception('Whisper preload failed — first stream may be slow')
    t1 = time.perf_counter()
    try:
        # Preloading is not required for Gemini Analyzer API context
        pass
    except Exception:
        logger.exception('LLM loading failed — first analysis may be slow')
    t2 = time.perf_counter()
    logger.info(
        'ML preload complete | whisper_s=%.2f | sbert_s=%.2f | total_s=%.2f',
        t1 - t0,
        t2 - t1,
        t2 - t0,
    )


def create_server(config: Settings) -> grpc.Server:
    """
    Create and configure the gRPC server.

    Args:
        config: Application settings

    Returns:
        Configured gRPC server instance
    """
    # Validate proto code exists
    if not validate_proto_code():
        raise RuntimeError("Failed to validate or generate proto code")

    # Expose Prometheus metrics endpoint (HTTP /metrics).
    # This is best-effort: if the dependency is missing, we just log and proceed.
    if config.metrics_enabled:
        try:
            from prometheus_client import start_http_server

            start_http_server(config.metrics_port)
            logger.info(
                '📈 Prometheus /metrics enabled | port=%s',
                config.metrics_port,
            )
        except Exception:
            logger.warning(
                'Prometheus metrics disabled (dependency missing?): port=%s',
                config.metrics_port,
            )

    # Create server with thread pool
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc_workers)
    )

    # Create services
    sliding_window_worker = SlidingWindowWorker(
        min_window_seconds=config.audio_buffer_min_window_seconds,
        min_interval_ms=config.audio_buffer_min_interval_ms,
    )
    audio_buffer_service = AudioBufferService(
        worker=sliding_window_worker,
        window_seconds=config.audio_buffer_window_seconds,
    )
    text_analysis_service = TextAnalysisService()
    service_jwt_provider: ServiceJwtProvider | None = None
    if config.grpc_feedback_enabled and config.grpc_feedback_wants_auto_jwt():
        assert config.backend_http_base_url
        assert config.service_bootstrap_key
        service_jwt_provider = ServiceJwtProvider(
            http_base_url=config.backend_http_base_url,
            bootstrap_key=config.service_bootstrap_key,
            ttl_seconds=config.service_token_mint_ttl_seconds,
            mint_retries=config.service_token_mint_retries,
            mint_backoff_seconds=config.service_token_mint_backoff_seconds,
        )
        service_jwt_provider.prewarm()

    backend_feedback_client = BackendFeedbackClient(
        service_url=config.grpc_feedback_url,
        enabled=config.grpc_feedback_enabled,
        timeout_seconds=config.grpc_feedback_timeout_seconds,
        service_token=config.grpc_feedback_service_token
        if service_jwt_provider is None
        else None,
        service_jwt_provider=service_jwt_provider,
    )
    publish_dispatcher = PublishDispatcher(
        backend_feedback_client.publish_feedback,
        max_queue_size=config.publish_queue_max_size,
        worker_threads=config.publish_worker_threads,
        max_event_age_ms=config.publish_max_age_ms,
        retry_limit=config.publish_retry_limit,
        retry_backoff_ms=config.publish_retry_backoff_ms,
    )

    # Inject publish_dispatcher into TextAnalysisService for deferred rate-limit dispatch
    if get_settings().llm_provider != 'ollama':
        text_analysis_service._publish_dispatcher = publish_dispatcher

    transcription_service: TranscriptionService | None = None
    if config.stt_provider == 'local':
        transcription_service = TranscriptionService(
            model_size=config.transcription_model_size,
            device=config.transcription_device,
            compute_type=config.transcription_compute_type,
            vad_filter=config.whisper_vad_filter,
            empty_diagnostic_no_vad=config.whisper_empty_diagnostic_no_vad,
            low_energy_dbfs_threshold=config.whisper_low_energy_dbfs,
            default_language=config.whisper_default_language,
            process_workers=config.stt_process_workers,
        )

    transcription_pipeline_service = TranscriptionPipelineService(
        transcription_service=transcription_service,
        text_analysis_service=text_analysis_service,
        publish_dispatcher=publish_dispatcher,
        default_language=config.whisper_default_language,
        partial_min_confidence=config.partial_min_confidence,
        feedback_allow_host_publish=config.feedback_allow_host_publish,
    )
    partial_coordinator: PartialTurnCoordinator | None = None
    if config.stt_provider == 'assemblyai' and config.partial_analysis_enabled:

        def _on_partial_ready(
            stream_key: str,
            chunk: object,
            extra: dict[str, object],
        ) -> None:
            from ..modules.text_analysis.types import TranscriptionChunk

            assert isinstance(chunk, TranscriptionChunk)
            trace_ctx = LatencyTraceContext(
                trace_id=make_feedback_trace_id(
                    chunk.meeting_id,
                    chunk.participant_id,
                    chunk.window_end_ms,
                ),
                meeting_id=chunk.meeting_id,
                participant_id=chunk.participant_id,
                window_end_ms=chunk.window_end_ms,
            )
            turn_open_ms = extra.get('turnOpenMs')
            log_assemblyai_partial_stable(
                logger,
                trace_ctx,
                stream_key=stream_key,
                transcript_chars=len(chunk.text),
                completeness_reason=str(extra.get('completenessReason') or ''),
                turn_open_ms=turn_open_ms if isinstance(turn_open_ms, int) else None,
            )
            transcription_pipeline_service.process_transcript(
                stream_key,
                chunk,
                transcript_source='partial',
                extra_meta=extra,
            )

        partial_coordinator = PartialTurnCoordinator(
            PartialTurnConfig(
                enabled=True,
                stable_ms=config.partial_stable_ms,
                word_stable_ms=config.partial_word_stable_ms,
                growth_window_ms=config.partial_growth_window_ms,
                min_words=config.partial_min_words,
                cooldown_ms=config.partial_cooldown_ms,
            ),
            _on_partial_ready,
        )
        transcription_pipeline_service._partial_coordinator = partial_coordinator

    streaming_stt_provider: AssemblyAiStreamingProvider | None = None
    if config.stt_provider == 'assemblyai':
        assert config.assemblyai_api_key
        streaming_stt_provider = AssemblyAiStreamingProvider(
            AssemblyAiStreamConfig(
                api_key=config.assemblyai_api_key,
                api_host=config.assemblyai_api_host,
                speech_model=config.assemblyai_speech_model,
                sample_rate=config.assemblyai_sample_rate,
                format_turns=config.assemblyai_format_turns,
                continuous_partials=config.assemblyai_continuous_partials,
                stream_idle_timeout_ms=config.assemblyai_stream_idle_timeout_ms,
                reconnect_limit=config.assemblyai_reconnect_limit,
                connect_timeout_seconds=config.assemblyai_connect_timeout_seconds,
                termination_timeout_seconds=config.assemblyai_termination_timeout_seconds,
                end_of_turn_confidence_threshold=(
                    config.assemblyai_end_of_turn_confidence_threshold
                ),
                min_turn_silence_ms=config.assemblyai_min_turn_silence_ms,
                max_turn_silence_ms=config.assemblyai_max_turn_silence_ms,
                vad_threshold=config.assemblyai_vad_threshold,
                keyterms_prompt=config.assemblyai_keyterms_prompt,
                tab_audio_vad_threshold=config.assemblyai_tab_audio_vad_threshold,
                tab_audio_max_turn_silence_ms=(
                    config.assemblyai_tab_audio_max_turn_silence_ms
                ),
            ),
            transcription_pipeline_service.process_transcript,
            partial_coordinator=partial_coordinator,
        )
    else:
        ready_window_dispatcher = ReadyWindowDispatcher(
            transcription_pipeline_service.process_window,
            max_queue_size=config.window_queue_max_size,
            worker_threads=config.window_worker_threads,
            max_age_ms=config.window_max_age_ms,
            low_priority_speech_ratio_below=config.window_low_priority_speech_ratio_below,
        )
        audio_buffer_service.register_window_callback(
            lambda sk, pcm, meta: ready_window_dispatcher.enqueue(sk, pcm, meta),
        )

    audio_service = AudioService(
        audio_buffer_service=audio_buffer_service,
        streaming_stt_provider=streaming_stt_provider,
    )
    servicer = AudioPipelineServicer(audio_service)

    # Register servicer
    audio_pipeline_pb2_grpc.add_AudioPipelineServiceServicer_to_server(
        servicer,
        server
    )

    logger.info(f"Servidor gRPC criado com {config.grpc_workers} workers")
    if config.stt_provider == 'assemblyai':
        logger.info(
            'STT config | provider=assemblyai | model=%s | api_host=%s | '
            'sample_rate=%s | format_turns=%s | continuous_partials=%s | '
            'idle_timeout_ms=%s | min_turn_silence_ms=%s | max_turn_silence_ms=%s | '
            'vad_threshold=%s | tab_audio_vad_threshold=%s | '
            'tab_audio_max_turn_silence_ms=%s | partial_analysis=%s | '
            'partial_stable_ms=%s | partial_min_confidence=%s | '
            'feedback_allow_host_publish=%s',
            config.assemblyai_speech_model,
            config.assemblyai_api_host,
            config.assemblyai_sample_rate,
            config.assemblyai_format_turns,
            config.assemblyai_continuous_partials,
            config.assemblyai_stream_idle_timeout_ms,
            config.assemblyai_min_turn_silence_ms,
            config.assemblyai_max_turn_silence_ms,
            config.assemblyai_vad_threshold,
            config.assemblyai_tab_audio_vad_threshold,
            config.assemblyai_tab_audio_max_turn_silence_ms,
            config.partial_analysis_enabled,
            config.partial_stable_ms,
            config.partial_min_confidence,
            config.feedback_allow_host_publish,
        )
    else:
        logger.info(
            'STT config | provider=local | STT_PROCESS_WORKERS=%s | '
            'WHISPER_VAD_FILTER=%s | WHISPER_EMPTY_DIAGNOSTIC_NO_VAD=%s | '
            'WHISPER_LOW_ENERGY_DBFS=%s | WHISPER_DEFAULT_LANGUAGE=%s',
            config.stt_process_workers,
            config.whisper_vad_filter,
            config.whisper_empty_diagnostic_no_vad,
            config.whisper_low_energy_dbfs,
            config.whisper_default_language,
        )
    logger.info('Gemini LLM Analyzer enabled')
    logger.info(
        'Window queue | WINDOW_QUEUE_MAX_SIZE=%s | WINDOW_WORKER_THREADS=%s | '
        'WINDOW_MAX_AGE_MS=%s | WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW=%s',
        config.window_queue_max_size,
        config.window_worker_threads,
        config.window_max_age_ms,
        config.window_low_priority_speech_ratio_below,
    )
    logger.info(
        'Backend feedback publish | GRPC_FEEDBACK_ENABLED=%s | GRPC_FEEDBACK_URL=%s | '
        'SERVICE_JWT_AUTO_MINT=%s | PUBLISH_QUEUE_MAX_SIZE=%s | PUBLISH_WORKER_THREADS=%s | '
        'PUBLISH_MAX_AGE_MS=%s',
        config.grpc_feedback_enabled,
        config.grpc_feedback_url,
        service_jwt_provider is not None,
        config.publish_queue_max_size,
        config.publish_worker_threads,
        config.publish_max_age_ms,
    )

    if config.preload_ml_models and transcription_service is not None:
        logger.info('PRELOAD_ML_MODELS=true — loading Whisper + embedding model...')
        _warmup_ml_models(transcription_service, text_analysis_service)
    elif config.stt_provider == 'assemblyai':
        logger.info('STT_PROVIDER=assemblyai — skipping local Whisper preload')
    else:
        logger.info('PRELOAD_ML_MODELS=false — models load on first use')

    # Attach runtime resources required for graceful shutdown.
    setattr(
        server,
        '_audio_pipeline_runtime',
        _ServerRuntime(
            text_analysis_service=text_analysis_service,
            publish_dispatcher=publish_dispatcher,
            backend_feedback_client=backend_feedback_client,
            transcription_service=transcription_service,
            streaming_stt_provider=streaming_stt_provider,
        ),
    )
    return server


def start_server(server: grpc.Server, config: Settings) -> None:
    """
    Start the gRPC server and wait for termination.

    Args:
        server: gRPC server instance
        config: Application settings
    """
    runtime = getattr(server, '_audio_pipeline_runtime', None)

    def _shutdown_runtime() -> None:
        if runtime is not None:
            runtime.shutdown()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Recebido sinal {signum}, encerrando servidor...")
        _shutdown_runtime()
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Listen on all interfaces (IPv4). [::] alone can miss IPv4-only clients on private mesh.
    listen_addr = f'0.0.0.0:{config.grpc_port}'
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f"🚀 Servidor gRPC iniciado em {listen_addr}")
    logger.info(f"📡 Aguardando streams de áudio...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("🛑 Servidor encerrado pelo usuário")
        _shutdown_runtime()
        server.stop(0)
