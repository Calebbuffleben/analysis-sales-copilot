"""Pipeline service tests for audio aggregate fast path."""

from __future__ import annotations

import unittest

from src.modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)


class _FakeTranscriptionService:
    pass


class _FakeTextAnalysisService:
    pass


class _FakePublishDispatcher:
    def __init__(self) -> None:
        self.events = []

    def enqueue(self, event) -> bool:
        self.events.append(event)
        return True


class TestTranscriptionPipelineService(unittest.TestCase):
    def test_enqueue_audio_aggregate_publishes_audio_only_event(self) -> None:
        publish_dispatcher = _FakePublishDispatcher()
        service = TranscriptionPipelineService(
            transcription_service=_FakeTranscriptionService(),  # type: ignore[arg-type]
            text_analysis_service=_FakeTextAnalysisService(),  # type: ignore[arg-type]
            publish_dispatcher=publish_dispatcher,
            default_language='pt',
        )
        window_pcm = (b'\x10\x00' * 1600)
        meta = {
            'meeting_id': 'meeting-1',
            'participant_id': 'participant-1',
            'track': 'tab-audio',
            'sample_rate': 16000,
            'channels': 1,
            'window_start_ms': 1000,
            'window_end_ms': 2000,
        }

        enqueued = service.enqueue_audio_aggregate(
            'meeting-1:participant-1:tab-audio',
            window_pcm,
            meta,
        )

        self.assertTrue(enqueued)
        self.assertEqual(len(publish_dispatcher.events), 1)
        event = publish_dispatcher.events[0]
        self.assertEqual(event.feedback_type, 'audio_metrics_ingress')
        self.assertEqual(event.analysis.analysis_mode, 'audio_only')
        self.assertTrue(event.analysis.signal_validity['audio_aggregate'])
        self.assertFalse(event.analysis.signal_validity['indecision_fast'])
        self.assertFalse(event.analysis.signal_validity['indecision_semantic'])


if __name__ == '__main__':
    unittest.main()
