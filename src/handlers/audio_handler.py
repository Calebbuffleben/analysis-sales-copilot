"""gRPC handler for audio streams."""

import logging
import os
import sys

import grpc

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'proto'))

import audio_pipeline_pb2
import audio_pipeline_pb2_grpc

from ..services.audio_service import AudioService

logger = logging.getLogger(__name__)

class AudioPipelineServicer(audio_pipeline_pb2_grpc.AudioPipelineServiceServicer):
    """gRPC servicer for receiving audio streams."""

    def __init__(self, audio_service: AudioService):
        """
        Initialize the audio pipeline servicer.

        Args:
            audio_service: AudioService instance for processing audio chunks
        """
        self.audio_service = audio_service

    def StreamAudio(self, request_iterator, context):
        """
        Receive a stream of audio chunks (client streaming).

        Args:
            request_iterator: Iterator of AudioChunk messages
            context: gRPC context

        Returns:
            StreamAudioResponse message
        """
        chunks_received = 0
        meeting_id = None
        participant_id = None
        track = None
        stream_started = False

        try:
            for chunk in request_iterator:
                chunks_received += 1

                # Start stream on first chunk
                if chunks_received == 1:
                    meeting_id = chunk.meeting_id
                    participant_id = chunk.participant_id
                    track = chunk.track
                    logger.info(
                        "🎙️ StreamAudio started | meetingId=%s | participantId=%s | track=%s",
                        meeting_id,
                        participant_id,
                        track,
                    )

                    self.audio_service.start_stream(
                        meeting_id=meeting_id,
                        participant_id=participant_id,
                        track=track,
                        sample_rate=chunk.sample_rate,
                        channels=chunk.channels
                    )
                    stream_started = True

                # Process chunk
                self.audio_service.process_chunk(
                    meeting_id=chunk.meeting_id,
                    participant_id=chunk.participant_id,
                    track=chunk.track,
                    wav_data=chunk.wav_data,
                    sequence=chunk.sequence,
                    timestamp_ms=chunk.timestamp_ms
                )

                # TODO: Integrate with the audio buffering pipeline.
                # The AudioService (and the audio_buffer module) should:
                # - Update the sliding-window buffer for this stream using this new chunk.
                # - Decide when there is enough context (N seconds) to trigger transcription.
                # - Forward the resulting transcript to text-analysis services
                #   (emotion, sales intelligence, etc.) and emit feedback events.

            # End stream if it was started
            if stream_started:
                self.audio_service.end_stream(
                    meeting_id=meeting_id,
                    participant_id=participant_id,
                    track=track
                )
                logger.info(
                    "✅ StreamAudio finished | meetingId=%s | participantId=%s | chunks=%s",
                    meeting_id,
                    participant_id,
                    chunks_received,
                )

            # TODO: Consider returning or logging high-level stream metrics
            # (e.g., total audio duration, number of processed windows, quality indicators).
            return audio_pipeline_pb2.StreamAudioResponse(
                success=True,
                message=f"Received {chunks_received} audio chunks",
                chunks_received=chunks_received
            )

        except grpc.RpcError:
            # Re-raise gRPC errors
            raise
        except Exception as e:
            logger.error(
                f"❌ Erro ao processar stream | meetingId={meeting_id} | "
                f"participantId={participant_id} | error={str(e)}",
                exc_info=True
            )

            # End stream if it was started
            if stream_started:
                try:
                    self.audio_service.end_stream(
                        meeting_id=meeting_id,
                        participant_id=participant_id,
                        track=track
                    )
                except Exception:
                    pass  # Ignore errors during cleanup

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing audio stream: {str(e)}")

            return audio_pipeline_pb2.StreamAudioResponse(
                success=False,
                message=f"Error: {str(e)}",
                chunks_received=chunks_received
            )

