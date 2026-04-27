#!/usr/bin/env python3
"""
Entry point for the Audio Pipeline Service.

This module initializes the application, sets up logging,
and starts the gRPC server.
"""

import logging
import os
import sys

# Add parent directory to path when running as script
# This allows imports to work both as module and as standalone script
if __name__ == '__main__':
    # Get the directory containing this file (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add parent directory (/app) to path so we can import src.*
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Use absolute imports (works in both cases)
from src.config import get_settings, setup_logging
from src.grpc_server.server import create_server, start_server

# When run as `python src/main.py`, __name__ is "__main__" — use a stable name in logs.
logger = logging.getLogger(
    "audio_pipeline" if __name__ == "__main__" else __name__,
)

def main():
    """Main entry point for the application."""
    try:
        # Load configuration
        config = get_settings()

        # Setup logging
        setup_logging()

        logger.info("Iniciando Audio Pipeline Service...")

        # Create gRPC server
        server = create_server(config)

        # Start server (blocks until termination)
        start_server(server, config)

    except KeyboardInterrupt:
        logger.info("Aplicação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro fatal ao iniciar aplicação: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
