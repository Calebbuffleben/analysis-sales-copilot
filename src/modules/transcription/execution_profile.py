from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DegradationLevel = Literal['L0', 'L1', 'L2', 'L3']
AnalysisMode = Literal['full_semantic', 'semantic_suppressed']


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
    semantic_pipeline_enabled: bool
    compute_category_transition: bool
    low_priority_speech_ratio_below: float
    analysis_mode: AnalysisMode
    signal_validity: dict[str, bool] = field(default_factory=dict)
    suppression_reasons: list[str] = field(default_factory=list)

