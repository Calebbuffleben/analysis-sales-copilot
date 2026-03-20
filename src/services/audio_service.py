"""Service for processing audio chunks."""

import logging
from typing import Optional

from ..modules.audio_buffer.service import AudioBufferService
from .stream_service import StreamService, StreamStats

logger = logging.getLogger(__name__)


class AudioService:
    """Service for processing audio chunks and managing audio streams."""

    def __init__(
        self,
        stream_service: Optional[StreamService] = None,
        audio_buffer_service: Optional[AudioBufferService] = None,
    ):
        """
        Initialize the audio service.

        Args:
            stream_service: Optional StreamService instance. Creates new one if not provided.
        """
        self.stream_service = stream_service or StreamService()
        self.audio_buffer_service = audio_buffer_service
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
        timestamp_ms: int
    ) -> None:
        """
        Process a single audio chunk.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier
            wav_data: WAV audio data
            sequence: Chunk sequence number
            timestamp_ms: Timestamp in milliseconds
        """
        chunk_size = len(wav_data)

        # Update stream statistics
        stats = self.stream_service.update_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            chunk_size=chunk_size
        )

        # Log chunk details
        logger.info(
            f"📥 Audio RECEIVED | meetingId={meeting_id} | "
            f"participantId={participant_id} | chunk={sequence} | "
            f"size={chunk_size}bytes | timestamp={timestamp_ms}ms"
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

        # TODO: Forward this chunk to the AudioBufferService so it can:
        # - Manter um buffer de janela deslizante por stream (meeting_id:participant_id:track)
        #   usando CircularBuffer.
        # - Notificar o SlidingWindowWorker (on_chunk_appended), que decidirá
        #   quando há janela suficiente para disparar o callback registrado.
        # - O callback, implementado por TranscriptionPipelineService, chamará
        #   apenas serviços internos do python-service (STT/NLP) e não fará
        #   comunicação direta com o backend.
        if self.audio_buffer_service and stats:
            self.audio_buffer_service.push(
                stream_key=stats.key,
                wav_data=wav_data,
                sample_rate=stats.sample_rate,
                channels=stats.channels,
                timestamp_ms=timestamp_ms,
                sequence=sequence,
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
        if self.audio_buffer_service:
            self.audio_buffer_service.end_stream(stream_key)

        return self.stream_service.end_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track
        )
