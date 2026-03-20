"""Service for managing audio streams."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Statistics for an audio stream."""

    meeting_id: str
    participant_id: str
    track: str
    sample_rate: int
    channels: int
    chunks_received: int = 0
    bytes_received: int = 0
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    @property
    def duration_seconds(self) -> float:
        """Get stream duration in seconds."""
        return time.time() - self.start_time

    @property
    def key(self) -> str:
        """Get unique stream key."""
        return f"{self.meeting_id}:{self.participant_id}:{self.track}"

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def add_chunk(self, chunk_size: int) -> None:
        """Add chunk statistics."""
        self.chunks_received += 1
        self.bytes_received += chunk_size
        self.update_activity()


class StreamService:
    """Service for managing active audio streams."""

    def __init__(self):
        """Initialize the stream service."""
        self._streams: Dict[str, StreamStats] = {}

    def start_stream(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        sample_rate: int,
        channels: int
    ) -> StreamStats:
        """
        Start tracking a new audio stream.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier
            sample_rate: Audio sample rate
            channels: Number of audio channels

        Returns:
            StreamStats instance for the new stream
        """
        key = f"{meeting_id}:{participant_id}:{track}"
        stats = StreamStats(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            sample_rate=sample_rate,
            channels=channels,
        )
        self._streams[key] = stats

        logger.info(
            f"🎤 Stream iniciado | meetingId={meeting_id} | "
            f"participantId={participant_id} | track={track} | "
            f"sampleRate={sample_rate}Hz | channels={channels}"
        )

        return stats

    def get_stream(self, meeting_id: str, participant_id: str, track: str) -> Optional[StreamStats]:
        """
        Get stream statistics.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier

        Returns:
            StreamStats if stream exists, None otherwise
        """
        key = f"{meeting_id}:{participant_id}:{track}"
        return self._streams.get(key)

    def update_stream(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        chunk_size: int
    ) -> Optional[StreamStats]:
        """
        Update stream with new chunk data.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier
            chunk_size: Size of the chunk in bytes

        Returns:
            Updated StreamStats if stream exists, None otherwise
        """
        stats = self.get_stream(meeting_id, participant_id, track)
        if stats:
            stats.add_chunk(chunk_size)
        return stats

    def end_stream(self, meeting_id: str, participant_id: str, track: str) -> Optional[StreamStats]:
        """
        End tracking of an audio stream.

        Args:
            meeting_id: Meeting identifier
            participant_id: Participant identifier
            track: Track identifier

        Returns:
            Final StreamStats if stream existed, None otherwise
        """
        key = f"{meeting_id}:{participant_id}:{track}"
        stats = self._streams.pop(key, None)

        if stats:
            logger.info(
                f"✅ Stream finalizado | meetingId={meeting_id} | "
                f"participantId={participant_id} | totalChunks={stats.chunks_received} | "
                f"totalBytes={stats.bytes_received} | duration={stats.duration_seconds:.2f}s"
            )

        return stats

    def cleanup_inactive_streams(self, timeout_seconds: float = 300.0) -> int:
        """
        Clean up streams that have been inactive for too long.

        Args:
            timeout_seconds: Seconds of inactivity before cleanup

        Returns:
            Number of streams cleaned up
        """
        now = time.time()
        inactive_keys = [
            key for key, stats in self._streams.items()
            if now - stats.last_activity > timeout_seconds
        ]

        for key in inactive_keys:
            stats = self._streams.pop(key)
            logger.warning(
                f"🧹 Stream removido por inatividade | meetingId={stats.meeting_id} | "
                f"participantId={stats.participant_id} | track={stats.track} | "
                f"inactiveFor={now - stats.last_activity:.2f}s"
            )

        return len(inactive_keys)

    def get_all_streams(self) -> Dict[str, StreamStats]:
        """Get all active streams."""
        return self._streams.copy()
