"""gRPC client used to publish canonical feedback events to the backend."""

from __future__ import annotations

import logging
import os
import sys
import time
from importlib import import_module
from typing import Any, Optional

import grpc

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
    ) -> None:
        self._service_url = service_url
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds
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
        )

        analysis = event.analysis
        request.analysis.embedding.extend(analysis.embedding)
        request.analysis.keywords.extend(analysis.keywords)
        if analysis.speech_act:
            request.analysis.speech_act = analysis.speech_act
        if analysis.sales_category:
            request.analysis.sales_category = analysis.sales_category
        if analysis.sales_category_confidence is not None:
            request.analysis.sales_category_confidence = analysis.sales_category_confidence
        if analysis.category_intensity is not None:
            request.analysis.category_intensity = analysis.category_intensity
        if analysis.category_ambiguity is not None:
            request.analysis.category_ambiguity = analysis.category_ambiguity
        request.analysis.category_flags.update(analysis.category_flags)
        request.analysis.conditional_keywords_detected.extend(
            analysis.conditional_keywords_detected,
        )
        if analysis.samples_count is not None:
            request.analysis.samples_count = analysis.samples_count
        if analysis.speech_count is not None:
            request.analysis.speech_count = analysis.speech_count
        if analysis.mean_rms_dbfs is not None:
            request.analysis.mean_rms_dbfs = analysis.mean_rms_dbfs
        if analysis.analysis_mode:
            request.analysis.analysis_mode = analysis.analysis_mode
        if analysis.degradation_level:
            request.analysis.degradation_level = analysis.degradation_level
        if analysis.signal_validity:
            request.analysis.signal_validity.update(analysis.signal_validity)
        if analysis.suppression_reasons:
            request.analysis.suppression_reasons.extend(analysis.suppression_reasons)

        if analysis.indecision_metrics:
            indecision_metrics = analysis.indecision_metrics
            request.analysis.indecision_metrics.conditional_language_score = float(
                indecision_metrics.get('conditional_language_score', 0.0),
            )
            request.analysis.indecision_metrics.postponement_likelihood = float(
                indecision_metrics.get('postponement_likelihood', 0.0),
            )

        try:
            t0 = time.perf_counter()
            self._stub.PublishFeedback(
                request,
                timeout=self._timeout_seconds,
            )
            t1 = time.perf_counter()
        except grpc.RpcError as exc:
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
        logger.info(
            '📨 Feedback published | meetingId=%s | participantId=%s | type=%s | '
            'transcript_chars=%s | window=[%s..%s]',
            event.meeting_id,
            event.participant_id,
            event.feedback_type,
            transcript_chars,
            event.window_start_ms,
            event.window_end_ms,
        )

        publish_grpc_ms = (t1 - t0) * 1000.0
        return publish_grpc_ms

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
        self._channel = grpc.insecure_channel(self._service_url)
        self._stub = self._feedback_ingestion_pb2_grpc.FeedbackIngestionServiceStub(
            self._channel,
        )

    def close(self) -> None:
        """Close the gRPC channel when the process is shutting down."""
        if self._channel is not None:
            self._channel.close()
