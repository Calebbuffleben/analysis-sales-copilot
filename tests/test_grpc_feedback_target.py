"""Cloud Run / local GRPC_FEEDBACK_URL normalization."""

from __future__ import annotations

import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.config.settings import Settings  # noqa: E402


def test_https_feedback_url_uses_tls_and_port_443() -> None:
    target, use_tls = Settings._normalize_grpc_target(
        'https://meet-backend-xyz.run.app',
    )
    assert target == 'meet-backend-xyz.run.app:443'
    assert use_tls is True


def test_plain_host_port_is_insecure() -> None:
    target, use_tls = Settings._normalize_grpc_target('backend:50052')
    assert target == 'backend:50052'
    assert use_tls is False


def test_port_443_infers_tls() -> None:
    target, use_tls = Settings._normalize_grpc_target('meet-backend.run.app:443')
    assert target == 'meet-backend.run.app:443'
    assert use_tls is True
