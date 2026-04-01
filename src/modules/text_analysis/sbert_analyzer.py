"""SBERT-based semantic analyzer used by TextAnalysisService."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SBertAnalyzer:
    """Provide embeddings and lightweight semantic classification."""

    DEFAULT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    STOPWORDS = {
        'a', 'o', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'e', 'em', 'no',
        'na', 'nos', 'nas', 'um', 'uma', 'uns', 'umas', 'para', 'por', 'com',
        'que', 'se', 'eu', 'você', 'voce', 'ele', 'ela', 'eles', 'elas', 'me',
        'te', 'é', 'ser', 'foi', 'vai', 'vou', 'está', 'esta', 'isso', 'isto',
    }
    CATEGORY_EXAMPLES: Dict[str, List[str]] = {
        'price_interest': [
            'quanto custa isso',
            'qual o preço',
            'qual o valor do serviço',
            'como funciona o investimento',
        ],
        'decision_signal': [
            'vamos fechar',
            'quero seguir em frente',
            'pode mandar a proposta',
            'podemos avançar',
        ],
        'conversation_stalling': [
            'vou pensar',
            'depois eu vejo',
            'não é o momento agora',
            'mais para frente a gente fala',
        ],
        'client_indecision': [
            'não sei ainda',
            'talvez funcione',
            'preciso avaliar melhor',
            'não tenho certeza',
        ],
        'solution_understood': [
            'agora ficou claro',
            'entendi a proposta',
            'faz sentido para mim',
            'agora eu compreendi',
        ],
    }

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._model_load_attempted = False
        self._category_embedding_cache: Dict[str, List[List[float]]] = {}

    def extract_keywords(self, text: str) -> list[str]:
        """Extract inexpensive keywords from text."""
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower())
        keywords: List[str] = []
        seen = set()
        for token in tokens:
            if len(token) < 3 or token in self.STOPWORDS or token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                keywords.append(token)
        return keywords[:12]

    def load_sbert_model(self) -> Optional[Any]:
        """Load the SBERT model lazily and cache it."""
        if self._model is not None:
            return self._model
        if self._model_load_attempted:
            return None

        self._model_load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                'sentence-transformers is not installed; embeddings will be disabled.',
            )
            return None

        self._model = SentenceTransformer(self._model_name)
        return self._model

    def generate_semantic_embedding(self, text: str) -> list[float]:
        """Generate a semantic embedding for the given text."""
        model = self.load_sbert_model()
        if model is None:
            return []

        vector = model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embedding1 = self.generate_semantic_embedding(text1)
        embedding2 = self.generate_semantic_embedding(text2)
        if not embedding1 or not embedding2:
            return 0.0
        return self._cosine_similarity(embedding1, embedding2)

    def analyze_semantics(self, text: str) -> dict:
        """Return base semantic information for the text."""
        embedding = self.generate_semantic_embedding(text)
        return {
            'embedding': embedding,
            'embedding_dimension': len(embedding),
            'keywords': self.extract_keywords(text),
        }

    def generate_semantic_flags(
        self,
        scores: Dict[str, float],
    ) -> dict[str, bool]:
        """Generate boolean flags that simplify downstream routing."""
        return {
            'price_window_open': scores.get('price_interest', 0.0) >= 0.55,
            'decision_signal_strong': scores.get('decision_signal', 0.0) >= 0.6,
            'conversation_stalling': scores.get('conversation_stalling', 0.0) >= 0.55,
            'solution_understood': scores.get('solution_understood', 0.0) >= 0.55,
            'client_indecision': scores.get('client_indecision', 0.0) >= 0.5,
        }

    def classify_categories(
        self,
        text: str,
        *,
        use_embeddings: bool = True,
    ) -> Tuple[Optional[str], float, Dict[str, float], float, float, Dict[str, bool]]:
        """Classify text into a sales-related semantic category."""
        scores = self._score_categories(text, use_embeddings=use_embeddings)
        if not scores:
            return None, 0.0, {}, 0.0, 0.0, {}

        ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_category, best_score = ordered_scores[0]
        second_best_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0

        confidence = max(0.0, min(1.0, best_score - second_best_score))
        ambiguity = max(0.0, min(1.0, 1.0 - confidence))
        intensity = max(0.0, min(1.0, best_score))
        flags = self.generate_semantic_flags(scores)

        if best_score < 0.35:
            return None, 0.0, scores, ambiguity, intensity, flags

        return best_category, confidence, scores, ambiguity, intensity, flags

    def _score_categories(self, text: str, *, use_embeddings: bool) -> Dict[str, float]:
        if not use_embeddings:
            return self._score_categories_heuristically(text)

        if self.load_sbert_model() is None:
            return self._score_categories_heuristically(text)

        embedding = self.generate_semantic_embedding(text)
        if not embedding:
            return self._score_categories_heuristically(text)

        scores: Dict[str, float] = {}
        for category, examples in self.CATEGORY_EXAMPLES.items():
            example_embeddings = self._get_example_embeddings(category, examples)
            if not example_embeddings:
                scores[category] = 0.0
                continue

            similarities = [
                self._cosine_similarity(embedding, example_embedding)
                for example_embedding in example_embeddings
            ]
            scores[category] = sum(similarities) / len(similarities)

        return scores

    def _score_categories_heuristically(self, text: str) -> Dict[str, float]:
        normalized_text = text.lower()
        scores: Dict[str, float] = {}

        for category, examples in self.CATEGORY_EXAMPLES.items():
            matches = sum(
                1
                for example in examples
                if any(token in normalized_text for token in example.split())
            )
            scores[category] = min(1.0, matches / max(len(examples), 1))

        return scores

    def _get_example_embeddings(
        self,
        category: str,
        examples: List[str],
    ) -> List[List[float]]:
        if category in self._category_embedding_cache:
            return self._category_embedding_cache[category]

        embeddings = [
            self.generate_semantic_embedding(example)
            for example in examples
        ]
        filtered_embeddings = [embedding for embedding in embeddings if embedding]
        self._category_embedding_cache[category] = filtered_embeddings
        return filtered_embeddings

    def _cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
