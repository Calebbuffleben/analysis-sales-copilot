"""Tests for bounded ready-window dispatcher."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from src.modules.transcription.ready_window_dispatcher import ReadyWindowDispatcher


class TestReadyWindowDispatcher(unittest.TestCase):
    def test_stale_at_enqueue_not_processed(self) -> None:
        process = MagicMock()
        d = ReadyWindowDispatcher(
            process,
            max_queue_size=4,
            worker_threads=1,
            max_age_ms=60_000,
            low_priority_speech_ratio_below=0.02,
        )
        now_ms = int(time.time() * 1000)
        old_end = now_ms - 120_000
        ok = d.enqueue(
            'm:p:t',
            b'\x00\x00',
            {
                'sample_rate': 16000,
                'channels': 1,
                'meeting_id': 'm',
                'participant_id': 'p',
                'track': 't',
                'window_start_ms': old_end - 10_000,
                'window_end_ms': old_end,
                'speech_ratio': 0.5,
            },
        )
        self.assertFalse(ok)
        time.sleep(0.05)
        process.assert_not_called()

    def test_processes_fresh_window(self) -> None:
        evt = threading.Event()
        received: list[tuple] = []

        def process(sk: str, pcm: bytes, meta: dict) -> None:
            received.append((sk, pcm, meta))
            evt.set()

        d = ReadyWindowDispatcher(
            process,
            max_queue_size=4,
            worker_threads=1,
            max_age_ms=60_000,
            low_priority_speech_ratio_below=0.02,
        )
        now_ms = int(time.time() * 1000)
        self.assertTrue(
            d.enqueue(
                'm:p:t',
                b'\x01\x00',
                {
                    'sample_rate': 16000,
                    'channels': 1,
                    'meeting_id': 'm',
                    'participant_id': 'p',
                    'track': 't',
                    'window_start_ms': now_ms - 10_000,
                    'window_end_ms': now_ms,
                    'speech_ratio': 0.4,
                },
            )
        )
        self.assertTrue(evt.wait(timeout=3.0))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], 'm:p:t')


if __name__ == '__main__':
    unittest.main()
