"""Tests for the gRPC feedback client auth/tenancy metadata contract.

The backend enforces that every call carries ``authorization: Bearer <token>``
and, for SERVICE-role tokens, a mandatory ``x-tenant-id`` header. These tests
pin that contract from the Python side so regressions are caught locally.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest


SERVICE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'),
)
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.modules.backend_feedback.grpc_feedback_client import BackendFeedbackClient  # noqa: E402
from src.modules.backend_feedback.types import BackendFeedbackEvent  # noqa: E402
from src.modules.text_analysis.types import TextAnalysisResult  # noqa: E402


def _make_event(tenant_id: str = 'tenant-42') -> BackendFeedbackEvent:
    return BackendFeedbackEvent(
        meeting_id='m1',
        participant_id='p1',
        participant_name=None,
        participant_role=None,
        feedback_type='direct',
        severity='info',
        ts_ms=0,
        window_start_ms=0,
        window_end_ms=1000,
        message='hello',
        transcript_text='transcript',
        transcript_confidence=0.9,
        analysis=TextAnalysisResult(
            direct_feedback='',
            conversation_state_json='{}',
            samples_count=None,
            speech_count=None,
            mean_rms_dbfs=None,
        ),
        tenant_id=tenant_id,
    )


@pytest.fixture(autouse=True)
def _stub_grpc_pb2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the real gRPC stub bootstrap — we only care about the call args."""

    def _no_init(self: BackendFeedbackClient) -> None:
        self._feedback_ingestion_pb2 = MagicMock()
        self._feedback_ingestion_pb2.PublishFeedbackRequest = MagicMock(
            side_effect=lambda **kwargs: MagicMock(**kwargs),
        )
        self._feedback_ingestion_pb2_grpc = MagicMock()
        self._stub = MagicMock()
        self._channel = MagicMock()

    monkeypatch.setattr(
        BackendFeedbackClient,
        '_initialize_stub',
        _no_init,
    )


def test_metadata_includes_bearer_and_tenant_id() -> None:
    client = BackendFeedbackClient(
        service_url='localhost:50052',
        enabled=True,
        service_token='svc-token-123',
    )
    client.publish_feedback(_make_event('tenant-42'))

    assert client._stub.PublishFeedback.call_count == 1  # type: ignore[attr-defined]
    kwargs = client._stub.PublishFeedback.call_args.kwargs  # type: ignore[attr-defined]
    metadata = dict(kwargs['metadata'])
    assert metadata['authorization'] == 'Bearer svc-token-123'
    assert metadata['x-tenant-id'] == 'tenant-42'


def test_metadata_omits_authorization_when_token_missing() -> None:
    client = BackendFeedbackClient(
        service_url='localhost:50052',
        enabled=True,
        service_token=None,
    )
    client.publish_feedback(_make_event('tenant-77'))
    metadata = dict(
        client._stub.PublishFeedback.call_args.kwargs['metadata'],  # type: ignore[attr-defined]
    )
    assert 'authorization' not in metadata
    assert metadata['x-tenant-id'] == 'tenant-77'


def test_publish_refuses_empty_tenant_id() -> None:
    client = BackendFeedbackClient(
        service_url='localhost:50052',
        enabled=True,
        service_token='svc-token-123',
    )
    with pytest.raises(ValueError):
        client.publish_feedback(_make_event(''))
    assert client._stub.PublishFeedback.call_count == 0  # type: ignore[attr-defined]


def test_disabled_client_does_not_call_stub() -> None:
    client = BackendFeedbackClient(
        service_url='localhost:50052',
        enabled=False,
        service_token='svc-token-123',
    )
    # Disabled clients should never initialize a stub, so publish is a no-op.
    result = client.publish_feedback(_make_event('tenant-42'))
    assert result is None
