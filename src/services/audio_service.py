"""Service for processing audio chunks."""

import logging
from typing import Optional, TYPE_CHECKING

from ..modules.audio_buffer.service import AudioBufferService
from .stream_service import StreamService, StreamStats

if TYPE_CHECKING:
    from ..modules.text_analysis.text_analysis_service import TextAnalysisService

logger = logging.getLogger(__name__)


class AudioService:
    """Service for processing audio chunks and managing audio streams."""

    def __init__(
        self,
        stream_service: Optional[StreamService] = None,
        audio_buffer_service: Optional[AudioBufferService] = None,
        text_analysis_service: Optional['TextAnalysisService'] = None,
    ):
        """
        Initialize the audio service.

        Args:
            stream_service: Optional StreamService instance. Creates new one if not provided.
            audio_buffer_service: Optional AudioBufferService instance.
            text_analysis_service: Optional TextAnalysisService for cleanup on meeting end.
        """
        self.stream_service = stream_service or StreamService()
        self.audio_buffer_service = audio_buffer_service
        self.text_analysis_service = text_analysis_service

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

        FIX #3: Automatically clear conversation state when meeting ends.
        This prevents stale state from persisting for 30min after meeting ends.

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

        stats = self.stream_service.end_stream(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track
        )

        # FIX #3: Clear conversation state when stream ends
        if self.text_analysis_service:
            cleared = self.text_analysis_service.clear_meeting_state(meeting_id)
            if cleared > 0:
                logger.info(
                    f"🧹 Cleared {cleared} conversation states for meeting {meeting_id}"
                )

        return stats
