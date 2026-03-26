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

from ..config.settings import Settings
from ..handlers.audio_handler import AudioPipelineServicer
from ..modules.audio_buffer.service import AudioBufferService
from ..modules.audio_buffer.sliding_worker import SlidingWindowWorker
from ..modules.backend_feedback.grpc_feedback_client import BackendFeedbackClient
from ..modules.backend_feedback.publish_dispatcher import PublishDispatcher
from ..modules.text_analysis.text_analysis_service import TextAnalysisService
from ..modules.transcription.degradation_controller import DegradationController
from ..modules.transcription.ready_window_dispatcher import ReadyWindowDispatcher
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
    """Load Whisper and SBERT before the first audio window (avoids cold-start lag)."""
    t0 = time.perf_counter()
    try:
        transcription_service.preload_model()
    except Exception:
        logger.exception('Whisper preload failed — first stream may be slow')
    t1 = time.perf_counter()
    try:
        text_analysis_service.ensure_model_loaded()
    except Exception:
        logger.exception('Sentence-transformers preload failed — first analysis may be slow')
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
    text_analysis_service = TextAnalysisService()
    backend_feedback_client = BackendFeedbackClient(
        service_url=config.grpc_feedback_url,
        enabled=config.grpc_feedback_enabled,
        timeout_seconds=config.grpc_feedback_timeout_seconds,
    )
    publish_dispatcher = PublishDispatcher(
        backend_feedback_client.publish_feedback,
        max_queue_size=config.publish_queue_max_size,
        worker_threads=config.publish_worker_threads,
        max_event_age_ms=config.publish_max_age_ms,
        retry_limit=config.publish_retry_limit,
        retry_backoff_ms=config.publish_retry_backoff_ms,
    )
    transcription_pipeline_service = TranscriptionPipelineService(
        transcription_service=transcription_service,
        text_analysis_service=text_analysis_service,
        publish_dispatcher=publish_dispatcher,
        default_language=config.whisper_default_language,
    )
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
    audio_service = AudioService(audio_buffer_service=audio_buffer_service)
    servicer = AudioPipelineServicer(audio_service)

    # Register servicer
    audio_pipeline_pb2_grpc.add_AudioPipelineServiceServicer_to_server(
        servicer,
        server
    )

    logger.info(f"Servidor gRPC criado com {config.grpc_workers} workers")
    logger.info(
        'STT config | STT_PROCESS_WORKERS=%s | WHISPER_VAD_FILTER=%s | '
        'WHISPER_EMPTY_DIAGNOSTIC_NO_VAD=%s | WHISPER_LOW_ENERGY_DBFS=%s | '
        'WHISPER_DEFAULT_LANGUAGE=%s',
        config.stt_process_workers,
        config.whisper_vad_filter,
        config.whisper_empty_diagnostic_no_vad,
        config.whisper_low_energy_dbfs,
        config.whisper_default_language,
    )
    logger.info(
        'Window queue | WINDOW_QUEUE_MAX_SIZE=%s | WINDOW_WORKER_THREADS=%s | '
        'WINDOW_MAX_AGE_MS=%s | WINDOW_LOW_PRIORITY_SPEECH_RATIO_BELOW=%s',
        config.window_queue_max_size,
        config.window_worker_threads,
        config.window_max_age_ms,
        config.window_low_priority_speech_ratio_below,
    )

    if config.preload_ml_models:
        logger.info('PRELOAD_ML_MODELS=true — loading Whisper + embedding model...')
        _warmup_ml_models(transcription_service, text_analysis_service)
    else:
        logger.info('PRELOAD_ML_MODELS=false — models load on first use')

    degradation_controller = DegradationController(
        scheduler=ready_window_dispatcher,
        pipeline_service=transcription_pipeline_service,
        publish_dispatcher=publish_dispatcher,
        base_low_priority_speech_ratio_below=config.window_low_priority_speech_ratio_below,
        degradation_enabled=config.degradation_enabled,
        eval_interval_ms=config.degradation_eval_interval_ms,
        l1_queue_age_ms=config.degradation_l1_queue_age_ms,
        l2_queue_age_ms=config.degradation_l2_queue_age_ms,
        l3_queue_age_ms=config.degradation_l3_queue_age_ms,
        hysteresis_factor=config.degradation_hysteresis_factor,
        publish_queue_l2_ratio=config.degradation_publish_queue_l2_ratio,
        publish_queue_l3_ratio=config.degradation_publish_queue_l3_ratio,
    )
    degradation_controller.start()

    return server


def start_server(server: grpc.Server, config: Settings) -> None:
    """
    Start the gRPC server and wait for termination.

    Args:
        server: gRPC server instance
        config: Application settings
    """
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Recebido sinal {signum}, encerrando servidor...")
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
        server.stop(0)
