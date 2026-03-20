"""Logging configuration."""

import logging
import sys
from typing import Optional

from .settings import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Optional log level override. If not provided, uses settings.
    """
    settings = get_settings()
    level = log_level or settings.log_level

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )

    # Set specific logger levels if needed
    logging.getLogger('grpc').setLevel(logging.WARNING)
