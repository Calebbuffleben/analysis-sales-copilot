"""Degradation controller stabilization tests."""

from __future__ import annotations

import unittest

from src.modules.transcription.degradation_controller import DegradationController


class _FakeScheduler:
    def __init__(self) -> None:
        self.oldest_pending_age_ms = 0
        self.pending_size = 0
        self.low_priority_speech_ratio_below = 0.0

    def get_oldest_pending_age_ms(self) -> int:
        return self.oldest_pending_age_ms

    def get_pending_size(self) -> int:
        return self.pending_size

    def set_low_priority_speech_ratio_below(self, value: float) -> None:
        self.low_priority_speech_ratio_below = value


class _FakePipelineService:
    def __init__(self) -> None:
        self.profiles: list = []

    def set_execution_profile(self, profile) -> None:
        self.profiles.append(profile)


class _FakePublishDispatcher:
    def __init__(self) -> None:
        self.queue_size = 0
        self.max_queue_size = 64

    def get_queue_size(self) -> int:
        return self.queue_size

    def get_max_queue_size(self) -> int:
        return self.max_queue_size


class TestDegradationController(unittest.TestCase):
    def test_recovery_is_stepwise_and_requires_consecutive_healthy_evals(self) -> None:
        scheduler = _FakeScheduler()
        pipeline = _FakePipelineService()
        publish_dispatcher = _FakePublishDispatcher()
        controller = DegradationController(
            scheduler=scheduler,
            pipeline_service=pipeline,
            publish_dispatcher=publish_dispatcher,
            base_low_priority_speech_ratio_below=0.02,
            recovery_consecutive_evals=3,
            min_level_hold_ms=0,
        )

        controller._apply_profile(controller._make_profile('L0'))
        scheduler.oldest_pending_age_ms = 2600
        scheduler.pending_size = 2
        controller._evaluate_and_apply()
        self.assertEqual(controller._current_level, 'L2')

        scheduler.oldest_pending_age_ms = 100
        scheduler.pending_size = 0
        controller._evaluate_and_apply()
        controller._evaluate_and_apply()
        self.assertEqual(controller._current_level, 'L2')

        controller._evaluate_and_apply()
        self.assertEqual(controller._current_level, 'L1')

        controller._evaluate_and_apply()
        controller._evaluate_and_apply()
        self.assertEqual(controller._current_level, 'L1')

        controller._evaluate_and_apply()
        self.assertEqual(controller._current_level, 'L0')

    def test_indecision_fast_path_survives_l2_but_not_l3(self) -> None:
        scheduler = _FakeScheduler()
        pipeline = _FakePipelineService()
        publish_dispatcher = _FakePublishDispatcher()
        controller = DegradationController(
            scheduler=scheduler,
            pipeline_service=pipeline,
            publish_dispatcher=publish_dispatcher,
            base_low_priority_speech_ratio_below=0.02,
        )

        l2 = controller._make_profile('L2')
        l3 = controller._make_profile('L3')

        self.assertTrue(l2.signal_validity['indecision_fast'])
        self.assertFalse(l2.signal_validity['indecision_semantic'])
        self.assertTrue(l2.signal_validity['audio_aggregate'])
        self.assertFalse(l3.signal_validity['indecision_fast'])
        self.assertFalse(l3.signal_validity['indecision_semantic'])
        self.assertTrue(l3.signal_validity['audio_aggregate'])

    def test_publish_pressure_is_capped_below_semantic_suppression(self) -> None:
        scheduler = _FakeScheduler()
        pipeline = _FakePipelineService()
        publish_dispatcher = _FakePublishDispatcher()
        publish_dispatcher.queue_size = publish_dispatcher.max_queue_size
        controller = DegradationController(
            scheduler=scheduler,
            pipeline_service=pipeline,
            publish_dispatcher=publish_dispatcher,
            base_low_priority_speech_ratio_below=0.02,
        )

        controller._apply_profile(controller._make_profile('L0'))
        controller._evaluate_and_apply()

        self.assertEqual(controller._current_level, 'L1')


if __name__ == '__main__':
    unittest.main()
