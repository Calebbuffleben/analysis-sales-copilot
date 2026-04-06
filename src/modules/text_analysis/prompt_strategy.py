"""A/B testing framework for LLM prompt strategies.

Allows testing different prompt versions in production without deploying code changes.
Uses hash-based assignment to ensure consistent experience per meeting.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Available prompt strategies for A/B testing."""
    
    V1_CURRENT = "current_prompt"
    V2_WITH_EXAMPLES = "few_shot_prompt"
    V3_CONFIDENCE_AWARE = "confidence_aware_prompt"


class PromptStrategySelector:
    """Selects which prompt strategy to use for a given meeting.
    
    Features:
    - Hash-based assignment (same meeting always gets same strategy)
    - Configurable traffic split
    - Easy to add new strategies
    """
    
    def __init__(
        self,
        strategy: PromptStrategy = PromptStrategy.V1_CURRENT,
        ab_test_enabled: bool = False,
        ab_test_percentage: int = 50,  # Percentage of traffic for B variant
    ):
        """
        Args:
            strategy: Default strategy to use
            ab_test_enabled: Whether A/B testing is enabled
            ab_test_percentage: Percentage of traffic for variant B (0-100)
        """
        self.default_strategy = strategy
        self.ab_test_enabled = ab_test_enabled
        self.ab_test_percentage = max(0, min(100, ab_test_percentage))
        
        logger.info(
            f"Prompt strategy selector initialized: "
            f"default={strategy.value}, "
            f"ab_test={ab_test_enabled} ({ab_test_percentage}% for B)"
        )
    
    def select_strategy(self, meeting_id: str) -> PromptStrategy:
        """Select prompt strategy for a meeting.
        
        Uses hash-based assignment to ensure consistency:
        - Same meeting_id always gets the same strategy
        - Traffic split is deterministic and even
        """
        if not self.ab_test_enabled:
            return self.default_strategy
        
        # Hash-based assignment
        hash_value = hash(meeting_id) % 100
        
        if hash_value < self.ab_test_percentage:
            selected = PromptStrategy.V2_WITH_EXAMPLES
        else:
            selected = self.default_strategy
        
        logger.debug(
            f"Strategy selected for meeting {meeting_id}: "
            f"{selected.value} (hash={hash_value})"
        )
        
        return selected
    
    def get_strategy_config(self, strategy: PromptStrategy) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        configs = {
            PromptStrategy.V1_CURRENT: {
                'name': 'Current Prompt',
                'temperature': 0.2,
                'has_examples': False,
                'has_confidence': False,
            },
            PromptStrategy.V2_WITH_EXAMPLES: {
                'name': 'Few-Shot Examples',
                'temperature': 0.2,
                'has_examples': True,
                'has_confidence': False,
            },
            PromptStrategy.V3_CONFIDENCE_AWARE: {
                'name': 'Confidence-Aware',
                'temperature': 0.3,
                'has_examples': True,
                'has_confidence': True,
            },
        }
        return configs.get(strategy, configs[PromptStrategy.V1_CURRENT])


# Global instance for easy import
_prompt_selector: Optional[PromptStrategySelector] = None


def get_prompt_strategy_selector() -> PromptStrategySelector:
    """Get or create the global prompt strategy selector."""
    global _prompt_selector
    if _prompt_selector is None:
        # Read from environment if available
        import os
        ab_enabled = os.getenv('LLM_AB_TEST_ENABLED', 'false').lower() == 'true'
        ab_percentage = int(os.getenv('LLM_AB_TEST_PERCENTAGE', '50'))
        
        _prompt_selector = PromptStrategySelector(
            strategy=PromptStrategy.V2_WITH_EXAMPLES,  # Default to improved prompt
            ab_test_enabled=ab_enabled,
            ab_test_percentage=ab_percentage,
        )
    
    return _prompt_selector


def reset_prompt_strategy_selector() -> None:
    """Reset the global selector (useful for testing)."""
    global _prompt_selector
    _prompt_selector = None
