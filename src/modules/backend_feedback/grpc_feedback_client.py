"""gRPC client used to publish canonical feedback events to the backend."""

from __future__ import annotations

import logging
import os
import sys
import time
from importlib import import_module
from typing import Any, Optional

import grpc

from ...feedback_trace import log_feedback_trace, make_feedback_trace_id
from .service_jwt_provider import ServiceJwtProvider
from .types import BackendFeedbackEvent
from ...metrics.realtime_metrics import FEEDBACK_PUBLISH_ERRORS_TOTAL

logger = logging.getLogger(__name__)


class BackendFeedbackClient:
    """Publish feedback events to the backend's dedicated gRPC ingress."""

    def __init__(
        self,
        service_url: str,
        enabled: bool = True,
        timeout_seconds: float = 5.0,
        service_token: Optional[str] = None,
        service_jwt_provider: Optional[ServiceJwtProvider] = None,
        use_tls: bool = False,
    ) -> None:
        self._service_url = service_url
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds
        self._use_tls = use_tls
        # Service-to-service JWT. The backend requires a Bearer token for all
        # gRPC ingress calls; this one is minted with role=SERVICE and is
        # permitted to operate cross-tenant provided x-tenant-id is passed.
        self._service_token = service_token
        self._jwt_provider = service_jwt_provider
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[Any] = None
        self._feedback_ingestion_pb2: Optional[Any] = None
        self._feedback_ingestion_pb2_grpc: Optional[Any] = None

        if self._enabled:
            self._initialize_stub()

    def publish_feedback(self, event: BackendFeedbackEvent) -> float | None:
        """Publish one canonical feedback event to the backend."""
        if self._enabled and self._stub is None:
            self._initialize_stub()

        if not self._enabled or self._stub is None:
            logger.debug(
                'Skipping feedback publish because backend feedback client is disabled.',
            )
            return

        if not event.tenant_id:
            logger.error(
                'Refusing to publish feedback with empty tenant_id | meetingId=%s',
                event.meeting_id,
            )
            FEEDBACK_PUBLISH_ERRORS_TOTAL.inc()
            raise ValueError('tenant_id is required for feedback publication')

        request = self._feedback_ingestion_pb2.PublishFeedbackRequest(
            meeting_id=event.meeting_id,
            participant_id=event.participant_id,
            participant_name=event.participant_name or '',
            participant_role=event.participant_role or '',
            feedback_type=event.feedback_type,
            severity=event.severity,
            ts_ms=event.ts_ms,
            window_start_ms=event.window_start_ms,
            window_end_ms=event.window_end_ms,
            message=event.message,
            transcript_text=event.transcript_text,
            transcript_confidence=event.transcript_confidence,
            tenant_id=event.tenant_id,
        )

        analysis = event.analysis
        request.analysis.direct_feedback = analysis.direct_feedback
        request.analysis.conversation_state_json = analysis.conversation_state_json

        if analysis.samples_count is not None:
            request.analysis.samples_count = analysis.samples_count
        if analysis.speech_count is not None:
            request.analysis.speech_count = analysis.speech_count
        if analysis.mean_rms_dbfs is not None:
            request.analysis.mean_rms_dbfs = analysis.mean_rms_dbfs
        hint = (analysis.playbook_hint_json or '').strip()
        if hint:
            request.analysis.playbook_hint_json = hint

        logger.info(
            '[Step 6] Enviando feedback gerado pelo Gemini via gRPC para o backend',
        )
        t0 = time.perf_counter()
        t1 = t0
        for attempt in range(2):
            metadata = self._build_call_metadata(event.tenant_id)
            try:
                t0 = time.perf_counter()
                self._stub.PublishFeedback(
                    request,
                    timeout=self._timeout_seconds,
                    metadata=metadata,
                )
                t1 = time.perf_counter()
                break
            except grpc.RpcError as exc:
                details = (exc.details() or '').lower()
                if (
                    attempt == 0
                    and self._jwt_provider is not None
                    and exc.code() == grpc.StatusCode.UNAUTHENTICATED
                    and 'expired' in details
                ):
                    logger.warning(
                        'Feedback publish JWT expired; refreshing service token and retrying once | '
                        'meetingId=%s',
                        event.meeting_id,
                    )
                    self._jwt_provider.invalidate()
                    continue
                FEEDBACK_PUBLISH_ERRORS_TOTAL.inc()
                logger.error(
                    '📨 Feedback publish failed (gRPC) | meetingId=%s | participantId=%s | '
                    'type=%s | code=%s | details=%s',
                    event.meeting_id,
                    event.participant_id,
                    event.feedback_type,
                    exc.code(),
                    exc.details(),
                )
                raise
            except Exception as exc:
                FEEDBACK_PUBLISH_ERRORS_TOTAL.inc()
                logger.error(
                    '📨 Feedback publish failed | meetingId=%s | participantId=%s | type=%s | %s',
                    event.meeting_id,
                    event.participant_id,
                    event.feedback_type,
                    exc,
                    exc_info=True,
                )
                raise

        transcript_chars = len(event.transcript_text or '')
        publish_grpc_ms = (t1 - t0) * 1000.0
        tid = make_feedback_trace_id(
            event.meeting_id,
            event.participant_id,
            int(event.window_end_ms),
        )
        log_feedback_trace(
            logger,
            logging.INFO,
            'python.publish',
            trace_id=tid,
            meeting_id=event.meeting_id,
            participant_id=event.participant_id,
            window_end_ms=int(event.window_end_ms),
            extra={
                'feedbackType': event.feedback_type,
                'grpcMs': round(publish_grpc_ms, 1),
                'transcriptChars': transcript_chars,
                'windowStartMs': int(event.window_start_ms),
            },
        )
        return publish_grpc_ms

    def _auth_token(self) -> Optional[str]:
        if self._jwt_provider is not None:
            return self._jwt_provider.get_token()
        return self._service_token

    def _build_call_metadata(self, tenant_id: str) -> tuple[tuple[str, str], ...]:
        """Build per-call gRPC metadata with bearer token + tenant hint.

        The backend enforces:
        - ``authorization: Bearer <service JWT>`` MUST be present.
        - ``x-tenant-id`` is MANDATORY for service tokens (role=SERVICE) and
          is used as the effective tenant for the call.
        """
        token = self._auth_token()
        if not token:
            logger.warning(
                'Publishing feedback without a service token — the backend will reject.',
            )
            return (('x-tenant-id', tenant_id),)
        return (
            ('authorization', f'Bearer {token}'),
            ('x-tenant-id', tenant_id),
        )

    def _initialize_stub(self) -> None:
        """Create the gRPC stub lazily after proto modules are available."""
        if self._stub is not None:
            return

        proto_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '..',
            'proto',
        )
        sys.path.insert(0, os.path.abspath(proto_dir))
        self._feedback_ingestion_pb2 = import_module('feedback_ingestion_pb2')
        self._feedback_ingestion_pb2_grpc = import_module('feedback_ingestion_pb2_grpc')
        if self._use_tls:
            # Cloud Run / public HTTPS edge terminates TLS; use system roots.
            self._channel = grpc.secure_channel(
                self._service_url,
                grpc.ssl_channel_credentials(),
            )
        else:
            self._channel = grpc.insecure_channel(self._service_url)
        self._stub = self._feedback_ingestion_pb2_grpc.FeedbackIngestionServiceStub(
            self._channel,
        )

    def close(self) -> None:
        """Close the gRPC channel when the process is shutting down."""
        if self._channel is not None:
            self._channel.close()
