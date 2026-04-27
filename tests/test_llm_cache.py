"""Unit tests for LLM response cache."""

import time
import pytest
from .llm_cache import SimpleTextCache


class TestSimpleTextCache:
    """Test lightweight LLM response cache."""

    def test_exact_match_cache_hit(self):
        """Test exact text match returns cached response."""
        cache = SimpleTextCache()
        text = "Achei caro comparado ao concorrente"
        response = {
            "direct_feedback": "Destaque diferenciais",
            "confidence": 0.9,
        }
        
        cache.put(text, response)
        result = cache.get(text)
        
        assert result is not None
        assert result["direct_feedback"] == "Destaque diferenciais"
        assert result["confidence"] == 0.9

    def test_cache_miss_for_new_text(self):
        """Test cache miss for unseen text."""
        cache = SimpleTextCache()
        cache.put("Text one", {"feedback": "Feedback 1"})
        
        result = cache.get("Text two")
        
        assert result is None

    def test_similar_text_cache_hit(self):
        """Test similar text above threshold returns cached response."""
        cache = SimpleTextCache(similarity_threshold=0.7)
        text1 = "Achei o preço muito alto para o que oferece"
        text2 = "O preço está alto para o que vocês oferecem"
        response = {"feedback": "Destaque ROI"}
        
        cache.put(text1, response)
        result = cache.get(text2)
        
        assert result is not None
        assert result["feedback"] == "Destaque ROI"

    def test_different_text_cache_miss(self):
        """Test very different text results in cache miss."""
        cache = SimpleTextCache(similarity_threshold=0.85)
        cache.put("Achei caro", {"feedback": "Feedback 1"})
        
        # Completely different text
        result = cache.get("Como funciona o próximo passo?")
        
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SimpleTextCache(max_size=3)
        
        cache.put("Text 1", {"feedback": "F1"})
        cache.put("Text 2", {"feedback": "F2"})
        cache.put("Text 3", {"feedback": "F3"})
        
        # Access Text 1 to make it recently used
        cache.get("Text 1")
        
        # Add Text 4 - should evict Text 2 (LRU)
        cache.put("Text 4", {"feedback": "F4"})
        
        assert cache.get("Text 1") is not None  # Recently used
        assert cache.get("Text 2") is None      # Evicted (LRU)
        assert cache.get("Text 3") is not None
        assert cache.get("Text 4") is not None

    def test_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = SimpleTextCache(ttl_seconds=1)
        cache.put("Text", {"feedback": "Feedback"})
        
        # Immediate hit
        assert cache.get("Text") is not None
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("Text") is None

    def test_normalization_removes_punctuation(self):
        """Test text normalization removes punctuation."""
        cache = SimpleTextCache(similarity_threshold=0.7)
        text1 = "Achei caro!!!"
        text2 = "Achei caro"
        response = {"feedback": "Feedback"}
        
        cache.put(text1, response)
        result = cache.get(text2)
        
        # Should match after normalization
        assert result is not None

    def test_normalization_handles_case(self):
        """Test text normalization ignores case."""
        cache = SimpleTextCache(similarity_threshold=0.7)
        text1 = "ACHEI CARO"
        text2 = "achei caro"
        response = {"feedback": "Feedback"}
        
        cache.put(text1, response)
        result = cache.get(text2)
        
        assert result is not None

    def test_normalization_replaces_numbers(self):
        """Test text normalization replaces numbers."""
        cache = SimpleTextCache(similarity_threshold=0.7)
        text1 = "Preciso de 5 unidades"
        text2 = "Preciso de 10 unidades"
        response = {"feedback": "Feedback"}
        
        cache.put(text1, response)
        result = cache.get(text2)
        
        # Should match after number normalization
        assert result is not None

    def test_cache_statistics(self):
        """Test cache hit/miss statistics."""
        cache = SimpleTextCache()
        
        cache.put("Text 1", {"feedback": "F1"})
        cache.put("Text 2", {"feedback": "F2"})
        
        cache.get("Text 1")  # Hit
        cache.get("Text 3")  # Miss
        cache.get("Text 2")  # Hit
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.666) < 0.01
        assert stats["size"] == 2

    def test_cache_clear(self):
        """Test cache clear resets everything."""
        cache = SimpleTextCache()
        cache.put("Text", {"feedback": "Feedback"})
        cache.get("Text")
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_empty_text_handling(self):
        """Test empty text is handled gracefully."""
        cache = SimpleTextCache()
        
        cache.put("", {"feedback": "Feedback"})
        result = cache.get("")
        
        assert result is not None

    def test_none_text_handling(self):
        """Test None text is handled gracefully."""
        cache = SimpleTextCache()
        
        # Should not crash
        cache.put(None, {"feedback": "Feedback"})
        result = cache.get(None)
        
        # Normalization converts None to ""
        assert result is not None
