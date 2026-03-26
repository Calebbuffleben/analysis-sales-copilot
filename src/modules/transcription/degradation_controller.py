from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .execution_profile import ExecutionProfile

logger = logging.getLogger(__name__)


class DegradationController:
    """
    Runtime degradation controller (control plane).

    It computes a degradation level from local queue pressure metrics and
    updates:
    - TranscriptionPipelineService execution profile
    - ReadyWindowDispatcher low-priority admission threshold

    This is intentionally deterministic and stepwise to avoid flapping.
    """

    def __init__(
        self,
        *,
        scheduler,
        pipeline_service,
        publish_dispatcher,
        base_low_priority_speech_ratio_below: float,
        degradation_enabled: bool = True,
        eval_interval_ms: int = 500,
        # downgrade thresholds (older windows => more degradation)
        l1_queue_age_ms: int = 1000,
        l2_queue_age_ms: int = 2500,
        l3_queue_age_ms: int = 5000,
        # hysteresis (upgrade requires stricter recency)
        hysteresis_factor: float = 0.7,
        # publish queue influence
        publish_queue_l2_ratio: float = 0.8,
        publish_queue_l3_ratio: float = 0.95,
    ) -> None:
        self._scheduler = scheduler
        self._pipeline_service = pipeline_service
        self._publish_dispatcher = publish_dispatcher
        self._base_low_priority_speech_ratio_below = base_low_priority_speech_ratio_below
        self._enabled = degradation_enabled
        self._eval_interval_ms = eval_interval_ms

        self._l1_queue_age_ms = l1_queue_age_ms
        self._l2_queue_age_ms = l2_queue_age_ms
        self._l3_queue_age_ms = l3_queue_age_ms
        self._hysteresis_factor = hysteresis_factor
        self._publish_queue_l2_ratio = publish_queue_l2_ratio
        self._publish_queue_l3_ratio = publish_queue_l3_ratio

        self._lock = threading.Lock()
        self._current_level: str = 'L0'
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            logger.info('DegradationController disabled')
            return
        if self._thread is not None:
            return

        with self._lock:
            self._current_level = 'L0'

        self._apply_profile(self._make_profile('L0'))
        self._thread = threading.Thread(
            target=self._run_loop,
            name='degradation-controller',
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._evaluate_and_apply()
            except Exception:
                logger.exception('DegradationController evaluation failed')
            self._stop.wait(timeout=self._eval_interval_ms / 1000.0)

    def _evaluate_and_apply(self) -> None:
        oldest_pending_age_ms = self._scheduler.get_oldest_pending_age_ms()
        pending_size = self._scheduler.get_pending_size()

        publish_q = self._publish_dispatcher.get_queue_size()
        publish_max = self._publish_dispatcher.get_max_queue_size()

        # If publish_q is high relative to its own capacity, we increase degradation
        publish_level = 'L0'
        if publish_max > 0:
            ratio = publish_q / publish_max
            if ratio >= self._publish_queue_l3_ratio:
                publish_level = 'L3'
            elif ratio >= self._publish_queue_l2_ratio:
                publish_level = 'L2'

        # queue-age based desired level
        desired_by_age = 'L0'
        if oldest_pending_age_ms >= self._l3_queue_age_ms:
            desired_by_age = 'L3'
        elif oldest_pending_age_ms >= self._l2_queue_age_ms:
            desired_by_age = 'L2'
        elif oldest_pending_age_ms >= self._l1_queue_age_ms:
            desired_by_age = 'L1'

        # combine: worst level wins
        level_order = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3}
        desired_level = desired_by_age
        if level_order[publish_level] > level_order[desired_level]:
            desired_level = publish_level

        current_level = self._current_level
        if desired_level == current_level:
            return

        # Stepwise hysteresis to avoid flapping
        if level_order[desired_level] > level_order[current_level]:
            # downgrade (more degradation) happens immediately
            self._apply_profile(self._make_profile(desired_level))
            with self._lock:
                self._current_level = desired_level
            logger.info(
                '[degradation] level downgrade current=%s desired=%s oldestPendingAgeMs=%s pendingSize=%s publishQ=%s publishMax=%s',
                current_level,
                desired_level,
                oldest_pending_age_ms,
                pending_size,
                publish_q,
                publish_max,
            )
            return

        # upgrade (less degradation) requires stricter recency
        # We allow upgrading stepwise, not instantly.
        target_lower_level = desired_level
        upgrade_allowed_age_ms = self._upgrade_allowed_age_ms(target_lower_level)
        if oldest_pending_age_ms < upgrade_allowed_age_ms:
            self._apply_profile(self._make_profile(target_lower_level))
            with self._lock:
                self._current_level = target_lower_level
            logger.info(
                '[degradation] level upgrade current=%s target=%s oldestPendingAgeMs=%s pendingSize=%s publishQ=%s publishMax=%s',
                current_level,
                target_lower_level,
                oldest_pending_age_ms,
                pending_size,
                publish_q,
                publish_max,
            )

    def _upgrade_allowed_age_ms(self, target_level: str) -> int:
        # Allow upgrades only when queue age is sufficiently below the
        # downgrade threshold for that target.
        if target_level == 'L0':
            # to go to L0, need to be below L1 threshold with hysteresis
            return int(self._l1_queue_age_ms * self._hysteresis_factor)
        if target_level == 'L1':
            return int(self._l2_queue_age_ms * self._hysteresis_factor)
        if target_level == 'L2':
            return int(self._l3_queue_age_ms * self._hysteresis_factor)
        return 0

    def _make_profile(self, level: str) -> ExecutionProfile:
        # Map degradation level to execution flags.
        if level == 'L0':
            return ExecutionProfile(
                level='L0',
                use_embeddings=True,
                compute_category_transition=True,
                low_priority_speech_ratio_below=self._base_low_priority_speech_ratio_below * 1.0,
            )
        if level == 'L1':
            return ExecutionProfile(
                level='L1',
                use_embeddings=True,
                compute_category_transition=False,
                low_priority_speech_ratio_below=self._base_low_priority_speech_ratio_below * 1.5,
            )
        if level == 'L2':
            return ExecutionProfile(
                level='L2',
                use_embeddings=False,
                compute_category_transition=False,
                low_priority_speech_ratio_below=self._base_low_priority_speech_ratio_below * 2.5,
            )
        # L3
        return ExecutionProfile(
            level='L3',
            use_embeddings=False,
            compute_category_transition=False,
            low_priority_speech_ratio_below=self._base_low_priority_speech_ratio_below * 4.0,
        )

    def _apply_profile(self, profile: ExecutionProfile) -> None:
        # Update scheduler admission threshold
        if hasattr(self._scheduler, 'set_low_priority_speech_ratio_below'):
            self._scheduler.set_low_priority_speech_ratio_below(
                float(profile.low_priority_speech_ratio_below),
            )
        # Update pipeline flags
        if hasattr(self._pipeline_service, 'set_execution_profile'):
            self._pipeline_service.set_execution_profile(profile)

