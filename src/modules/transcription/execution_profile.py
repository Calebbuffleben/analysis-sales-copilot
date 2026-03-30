from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DegradationLevel = Literal['L0', 'L1', 'L2', 'L3']


@dataclass(frozen=True)
class ExecutionProfile:
    """
    Concrete execution flags for each degradation level.

    This makes degradation deterministic:
    the pipeline does not branch on ad-hoc booleans scattered in the codebase;
    it branches only on this profile object.
    """

    level: DegradationLevel
    use_embeddings: bool
    compute_category_transition: bool
    low_priority_speech_ratio_below: float

