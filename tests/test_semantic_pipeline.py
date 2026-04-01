"""Tests for SemanticPipeline (embedding toggle and category path)."""

from __future__ import annotations

import unittest

from src.modules.text_analysis.semantic_pipeline import SemanticPipeline
from src.modules.text_analysis.sbert_analyzer import SBertAnalyzer


class TestSemanticPipeline(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
