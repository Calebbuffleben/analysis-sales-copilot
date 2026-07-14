"""Service for processing audio chunks."""

import logging
from typing import Optional

from ..modules.audio_buffer.service import AudioBufferService
from ..modules.transcription.assemblyai_streaming_provider import (
    AssemblyAiStreamingProvider,
)
from .stream_service import StreamService, StreamStats

logger = logging.getLogger(__name__)


class AudioService:
    """Service for processing audio chunks and managing audio streams."""

    def __init__(
        self,
        stream_service: Optional[StreamService] = None,
        audio_buffer_service: Optional[AudioBufferService] = None,
        streaming_stt_provider: Optional[AssemblyAiStreamingProvider] = None,
    ):
        """
        Initialize the audio service.

        Args:
            stream_service: Optional StreamService instance. Creates new one if not provided.
        """
        self.stream_service = stream_service or StreamService()
        self.audio_buffer_service = audio_buffer_service
        self.streaming_stt_provider = streaming_stt_provider
        # TODO: Inject here, via dependency injection or factory:
        # - SlidingWindowWorker: já encapsulado dentro de AudioBufferService.
        # - TranscriptionPipelineService: serviço de nível mais alto que registra
        #   um callback no SlidingWindowWorker e orquestra STT + análise de texto.

    def start_stream(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        sample_rate: int,
        channels: int
    ) -> StreamStats:
        """
        Initialize a new audio stream.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier
            sample_rate: Audio sample rate
            channels: Number of audio channels

        Returns:
            StreamStats instance for the new stream
        """
        return self.stream_service.start_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            sample_rate=sample_rate,
            channels=channels
        )

    def process_chunk(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        wav_data: bytes,
        sequence: int,
        timestamp_ms: int,
        tenant_id: str = '',
        participant_role: str = '',
        acoustic_class: str = '',
        seller_room_id: str = '',
        matched_seller_id: str = '',
        correlation_confidence: float = 0.0,
    ) -> None:
        """
        Process a single audio chunk.
        """
        chunk_size = len(wav_data)

        # Update stream statistics
        stats = self.stream_service.update_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            chunk_size=chunk_size
        )

        # Log statistics every 100 chunks
        if stats and stats.chunks_received % 100 == 0:
            logger.info(
                f"📊 Audio STATS | meetingId={meeting_id} | "
                f"participantId={participant_id} | "
                f"chunks={stats.chunks_received} | "
                f"bytes={stats.bytes_received} | "
                f"duration={stats.duration_seconds:.2f}s"
            )

        if self.audio_buffer_service and stats:
            self.audio_buffer_service.push(
                stream_key=stats.key,
                wav_data=wav_data,
                sample_rate=stats.sample_rate,
                channels=stats.channels,
                timestamp_ms=timestamp_ms,
                sequence=sequence,
                tenant_id=tenant_id,
                participant_role=participant_role,
                acoustic_class=acoustic_class,
                seller_room_id=seller_room_id,
                matched_seller_id=matched_seller_id,
                correlation_confidence=correlation_confidence,
            )
        if self.streaming_stt_provider and stats:
            try:
                self.streaming_stt_provider.send_audio(
                    stats.key,
                    wav_data,
                    {
                        'stream_key': stats.key,
                        'meeting_id': meeting_id,
                        'participant_id': participant_id,
                        'track': track,
                        'sample_rate': stats.sample_rate,
                        'channels': stats.channels,
                        'timestamp_ms': timestamp_ms,
                        'sequence': sequence,
                        'tenant_id': tenant_id,
                        'participant_role': participant_role,
                        'acoustic_class': acoustic_class,
                        'seller_room_id': seller_room_id,
                        'matched_seller_id': matched_seller_id,
                        'correlation_confidence': correlation_confidence,
                    },
                )
            except Exception as exc:
                logger.exception(
                    'AssemblyAI streaming send failed | stream_key=%s | error=%s',
                    stats.key,
                    exc,
                )

    def end_stream(
        self,
        meeting_id: str,
        participant_id: str,
        track: str
    ) -> Optional[StreamStats]:
        """
        Finalize an audio stream.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier

        Returns:
            Final StreamStats if stream existed, None otherwise
        """
        stream_key = f"{meeting_id}:{participant_id}:{track}"
        if self.streaming_stt_provider:
            self.streaming_stt_provider.end_stream(stream_key)
        if self.audio_buffer_service:
            self.audio_buffer_service.end_stream(stream_key)

        return self.stream_service.end_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track
        )
