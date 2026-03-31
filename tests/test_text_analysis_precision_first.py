"""Precision-first degradation tests for text analysis."""

from __future__ import annotations

import unittest

from src.modules.text_analysis.text_analysis_service import TextAnalysisService
from src.modules.text_analysis.types import TranscriptionChunk
from src.modules.transcription.execution_profile import ExecutionProfile


class TestTextAnalysisPrecisionFirst(unittest.TestCase):
    def test_semantic_outputs_are_suppressed_when_signal_is_invalid(self) -> None:
        service = TextAnalysisService()
        chunk = TranscriptionChunk(
            meeting_id='meeting-1',
            participant_id='participant-1',
            track='mixed',
            text='Nao sei ainda, preciso avaliar melhor.',
            confidence=0.9,
            timestamp_ms=1_711_888_000_000,
            window_start_ms=1_711_887_995_000,
            window_end_ms=1_711_888_000_000,
        )
        profile = ExecutionProfile(
            level='L2',
            use_embeddings=False,
            semantic_pipeline_enabled=False,
            compute_category_transition=False,
            low_priority_speech_ratio_below=0.03,
            analysis_mode='semantic_suppressed',
            signal_validity={
                'indecision_fast': True,
                'indecision_semantic': False,
                'audio_aggregate': True,
            },
            suppression_reasons=[
                'indecision_semantic_suppressed_by_degradation',
            ],
        )

        out = service.analyze(chunk, execution_profile=profile)

        self.assertIsNone(out.sales_category)
        self.assertEqual(out.category_flags, {})
        self.assertEqual(out.analysis_mode, 'semantic_suppressed')
        self.assertEqual(out.degradation_level, 'L2')
        self.assertEqual(out.signal_validity['indecision_fast'], True)
        self.assertEqual(out.signal_validity['indecision_semantic'], False)
        self.assertIn(
            'indecision_semantic_suppressed_by_degradation',
            out.suppression_reasons,
        )


if __name__ == "__main__":
    unittest.main()
