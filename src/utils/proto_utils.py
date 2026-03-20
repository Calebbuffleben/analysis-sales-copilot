"""Utilities for Protocol Buffer code generation."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_proto_files(proto_dir: str, proto_file: str = 'audio_pipeline.proto') -> Tuple[bool, Optional[str]]:
    """
    Validate that proto files exist and are accessible.

    Args:
        proto_dir: Directory containing proto files
        proto_file: Name of the proto file to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    proto_path = Path(proto_dir) / proto_file

    if not proto_path.exists():
        error_msg = f"Arquivo .proto não encontrado: {proto_path}"
        logger.error(error_msg)
        return False, error_msg

    if not proto_path.is_file():
        error_msg = f"Caminho .proto não é um arquivo: {proto_path}"
        logger.error(error_msg)
        return False, error_msg

    return True, None


def validate_proto_file_list(
    proto_dir: str,
    proto_files: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    """Validate that all required proto files exist."""
    for proto_file in proto_files:
        is_valid, error_msg = validate_proto_files(proto_dir, proto_file)
        if not is_valid:
            return False, error_msg
    return True, None


def generate_proto_code(
    proto_dir: Optional[str] = None,
    proto_file: str = 'audio_pipeline.proto',
    output_dir: Optional[str] = None
) -> bool:
    """
    Generate Python code from Protocol Buffer definition.

    Args:
        proto_dir: Directory containing proto files. If None, uses default location.
        proto_file: Name of the proto file to process
        output_dir: Directory to output generated files. If None, uses proto_dir.

    Returns:
        True if generation succeeded, False otherwise
    """
    # Determine proto directory
    if proto_dir is None:
        proto_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')

    proto_dir = os.path.abspath(proto_dir)
    proto_path = os.path.join(proto_dir, proto_file)

    # Validate proto file exists
    is_valid, error_msg = validate_proto_files(proto_dir, proto_file)
    if not is_valid:
        return False

    # Use proto_dir as output if not specified
    if output_dir is None:
        output_dir = proto_dir

    output_dir = os.path.abspath(output_dir)

    try:
        logger.info(f"Gerando código gRPC a partir de {proto_path}...")
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'grpc_tools.protoc',
                f'--proto_path={proto_dir}',
                f'--python_out={output_dir}',
                f'--grpc_python_out={output_dir}',
                proto_path
            ],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.debug(f"Proto generation output: {result.stdout}")

        logger.info("Código gRPC gerado com sucesso")
        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"Erro ao gerar código gRPC: {e}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr}"
        logger.error(error_msg)
        return False
    except Exception as e:
        logger.error(f"Erro inesperado ao gerar código gRPC: {e}", exc_info=True)
        return False


def generate_proto_code_batch(
    proto_dir: Optional[str] = None,
    proto_files: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
) -> bool:
    """Generate Python code for multiple proto files."""
    if proto_dir is None:
        proto_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')

    proto_dir = os.path.abspath(proto_dir)
    files = list(proto_files or ['audio_pipeline.proto'])

    is_valid, error_msg = validate_proto_file_list(proto_dir, files)
    if not is_valid:
        logger.error(error_msg)
        return False

    for proto_file in files:
        if not generate_proto_code(
            proto_dir=proto_dir,
            proto_file=proto_file,
            output_dir=output_dir,
        ):
            return False

    return True
