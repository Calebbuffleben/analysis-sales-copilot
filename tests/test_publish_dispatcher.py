"""Tests for PublishDispatcher (bounded, non-blocking publish scheduling)."""

from __future__ import annotations

import threading
import time
import unittest

from src.modules.backend_feedback.publish_dispatcher import PublishDispatcher
from src.modules.backend_feedback.types import BackendFeedbackEvent
from src.modules.text_analysis.types import TextAnalysisResult


class TestPublishDispatcher(unittest.TestCase):
    def test_enqueue_is_accepted_and_publish_runs_in_background(self) -> None:
        published_evt = threading.Event()

        def publish_fn(event: BackendFeedbackEvent):
            # Simulate small network delay but do not block enqueue.
            time.sleep(0.05)
            published_evt.set()
            return 12.3

        d = PublishDispatcher(
            publish_fn,
            max_queue_size=2,
            worker_threads=1,
            max_event_age_ms=10_000,
            retry_limit=0,
        )

        ok = d.enqueue(
            BackendFeedbackEvent(
                meeting_id='m',
                participant_id='p',
                participant_name=None,
                participant_role=None,
                feedback_type='text_analysis_ingress',
                severity='info',
                ts_ms=int(time.time() * 1000),
                window_start_ms=0,
                window_end_ms=int(time.time() * 1000),
                message='x',
                transcript_text='t',
                transcript_confidence=0.9,
                analysis=TextAnalysisResult(),
            )
        )
        self.assertTrue(ok)
        self.assertTrue(published_evt.wait(timeout=2.0))

    def test_enqueue_drops_when_queue_full(self) -> None:
        # Hold the single worker so the queue fills up.
        gate_evt = threading.Event()

        publish_calls = 0

        def publish_fn(event: BackendFeedbackEvent):
            nonlocal publish_calls
            publish_calls += 1
            gate_evt.wait(timeout=2.0)
            return 1.0

        d = PublishDispatcher(
            publish_fn,
            max_queue_size=1,
            worker_threads=1,
            max_event_age_ms=10_000,
            retry_limit=0,
        )

        now = int(time.time() * 1000)
        e1 = BackendFeedbackEvent(
            meeting_id='m',
            participant_id='p',
            participant_name=None,
            participant_role=None,
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=now,
            window_start_ms=0,
            window_end_ms=now,
            message='x',
            transcript_text='t',
            transcript_confidence=0.9,
            analysis=TextAnalysisResult(),
        )
        e2 = BackendFeedbackEvent(
            meeting_id='m',
            participant_id='p',
            participant_name=None,
            participant_role=None,
            feedback_type='text_analysis_ingress',
            severity='info',
            ts_ms=now + 1,
            window_start_ms=0,
            window_end_ms=now + 1,
            message='x',
            transcript_text='t2',
            transcript_confidence=0.9,
            analysis=TextAnalysisResult(),
        )

        self.assertTrue(d.enqueue(e1))
        # Queue size is 1 and worker is blocked; second should drop.
        self.assertFalse(d.enqueue(e2))

        gate_evt.set()
        # allow background to finish cleanly
        time.sleep(0.1)

        self.assertGreaterEqual(publish_calls, 1)


if __name__ == '__main__':
    unittest.main()

