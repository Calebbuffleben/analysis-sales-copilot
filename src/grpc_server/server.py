"""gRPC server setup and initialization."""

import logging
import os
import signal
import sys
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
from ..modules.text_analysis.text_analysis_service import TextAnalysisService
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
    )
    text_analysis_service = TextAnalysisService()
    backend_feedback_client = BackendFeedbackClient(
        service_url=config.grpc_feedback_url,
        enabled=config.grpc_feedback_enabled,
        timeout_seconds=config.grpc_feedback_timeout_seconds,
    )
    transcription_pipeline_service = TranscriptionPipelineService(
        transcription_service=transcription_service,
        text_analysis_service=text_analysis_service,
        backend_feedback_client=backend_feedback_client,
        default_language=config.whisper_default_language,
    )
    audio_buffer_service.register_window_callback(
        transcription_pipeline_service._on_window_ready,
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
        'STT config | WHISPER_VAD_FILTER=%s | WHISPER_EMPTY_DIAGNOSTIC_NO_VAD=%s | '
        'WHISPER_LOW_ENERGY_DBFS=%s | WHISPER_DEFAULT_LANGUAGE=%s',
        config.whisper_vad_filter,
        config.whisper_empty_diagnostic_no_vad,
        config.whisper_low_energy_dbfs,
        config.whisper_default_language,
    )

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
