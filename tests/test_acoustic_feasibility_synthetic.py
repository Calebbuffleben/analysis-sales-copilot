"""Synthetic corpus and gate evaluation for Phase 0."""

from src.modules.acoustic_fingerprint.metrics import evaluate_corpus
from src.modules.acoustic_fingerprint.synthetic_corpus import default_synthetic_sessions


def test_synthetic_corpus_runs_end_to_end() -> None:
    report = evaluate_corpus(default_synthetic_sessions())
    assert report.sessions == 3
    assert report.windows >= 3
    assert 0.0 <= report.seller_tpr <= 1.0
    assert 0.0 <= report.routing_accuracy <= 1.0


def test_synthetic_corpus_passes_phase0_gates() -> None:
    report = evaluate_corpus(default_synthetic_sessions())
    assert report.passed, report.failures
