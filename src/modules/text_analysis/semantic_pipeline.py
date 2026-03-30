"""Semantic pipeline backed by the SBERT analyzer."""

from __future__ import annotations

from .sbert_analyzer import SBertAnalyzer


class SemanticPipeline:
    """Run the semantic analysis steps needed by TextAnalysisService."""

    def __init__(self, sbert_analyzer: SBertAnalyzer):
        self.sbert_analyzer = sbert_analyzer

    def run(self, text: str, *, use_embeddings: bool = True) -> dict:
        """Run semantic analysis and return normalized fields.

        When `use_embeddings` is False, we force heuristic category scoring and
        never generate semantic embeddings (saves CPU/GPU).
        """
        category, confidence, _, ambiguity, intensity, flags = (
            self.sbert_analyzer.classify_categories(
                text,
                use_embeddings=use_embeddings,
            )
        )

        embedding = (
            self.sbert_analyzer.generate_semantic_embedding(text)
            if use_embeddings
            else []
        )

        return {
            'embedding': embedding,
            'keywords': self.sbert_analyzer.extract_keywords(text),
            'sales_category': category,
            'sales_category_confidence': confidence if category else None,
            'category_intensity': intensity if category else None,
            'category_ambiguity': ambiguity if category else None,
            'category_flags': flags,
        }