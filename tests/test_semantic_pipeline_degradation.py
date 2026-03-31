"""Degradation behavior tests for semantic pipeline."""

from __future__ import annotations

import unittest

from src.modules.text_analysis.semantic_pipeline import SemanticPipeline
from src.modules.text_analysis.sbert_analyzer import SBertAnalyzer


class _CountingAnalyzer(SBertAnalyzer):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_calls = 0

    def load_sbert_model(self):  # type: ignore[override]
        return object()

    def generate_semantic_embedding(self, text: str) -> list[float]:  # type: ignore[override]
        self.embedding_calls += 1
        return [1.0, 0.0]

    def _get_example_embeddings(self, category: str, examples: list[str]) -> list[list[float]]:  # type: ignore[override]
        return [[1.0, 0.0] for _ in examples]


class TestSemanticPipelineDegradation(unittest.TestCase):
    def test_use_embeddings_false_returns_empty_embedding(self) -> None:
        pipeline = SemanticPipeline(SBertAnalyzer())

        out = pipeline.run(
            "Eu não sei ainda qual o preço disso.",
            use_embeddings=False,
        )

        self.assertEqual(out["embedding"], [])
        self.assertIn("keywords", out)
        self.assertIn("sales_category", out)
        self.assertIn("category_flags", out)

    def test_use_embeddings_true_encodes_text_only_once(self) -> None:
        analyzer = _CountingAnalyzer()
        pipeline = SemanticPipeline(analyzer)

        out = pipeline.run(
            "Eu não sei ainda qual o preço disso.",
            use_embeddings=True,
            include_payload_embedding=False,
        )

        self.assertEqual(analyzer.embedding_calls, 1)
        self.assertEqual(out["embedding"], [])
        self.assertIn("sales_category", out)


if __name__ == "__main__":
    unittest.main()

