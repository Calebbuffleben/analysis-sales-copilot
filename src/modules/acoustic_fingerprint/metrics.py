"""Evaluation metrics and gate checks for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AcousticFingerprintConfig
from .fingerprint_correlator import FingerprintCorrelator
from .fingerprint_generator import FingerprintGenerator
from .types import AcousticClass, CorpusSession, CorrelationResult


@dataclass
class WindowEvaluation:
    session_id: str
    scenario: str
    start_ms: int
    end_ms: int
    ground_truth: AcousticClass
    predicted: AcousticClass
    confidence: float
    matched_seller_id: str | None


@dataclass
class AcousticEvaluationReport:
    sessions: int
    windows: int
    seller_tpr: float
    customer_false_seller_fpr: float
    seller_false_customer_rate: float
    unknown_rate: float
    routing_accuracy: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: list[WindowEvaluation] = field(default_factory=list)


def _dominant_prediction(results: list[CorrelationResult]) -> AcousticClass:
    if not results:
        return 'unknown'
    counts = {'seller': 0, 'customer': 0, 'unknown': 0}
    for result in results:
        counts[result.acoustic_class] += 1
    return max(counts, key=counts.get)


def evaluate_session(
    session: CorpusSession,
    *,
    config: AcousticFingerprintConfig | None = None,
) -> list[WindowEvaluation]:
    cfg = config or AcousticFingerprintConfig()
    generator = FingerprintGenerator(cfg)
    correlator = FingerprintCorrelator(cfg, generator=generator)
    remote_fps = generator.fingerprint_stream(
        session.mic_pcm,
        user_id=session.seller_user_id,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
        channels=session.channels,
    )
    from .fingerprint_generator import pcm16_to_float32

    samples = pcm16_to_float32(session.loopback_pcm, session.channels)
    windows = list(generator.iter_windows(samples))
    results = correlator.correlate_stream(
        session.loopback_pcm,
        remote_fingerprints=remote_fps,
        seller_room_id=session.seller_room_id,
        meeting_id=session.meeting_id,
        channels=session.channels,
        simulated_lag_ms=session.simulated_lag_ms,
    )
    timed_results = [
        (start_ms, end_ms, results[idx] if idx < len(results) else None)
        for idx, (start_ms, end_ms, _window) in enumerate(windows)
    ]

    evaluations: list[WindowEvaluation] = []
    for label in session.labels:
        window_results = [
            result
            for start_ms, end_ms, result in timed_results
            if result is not None and start_ms >= label.start_ms and end_ms <= label.end_ms
        ]
        predicted = _dominant_prediction(window_results)
        confidence = max((r.confidence for r in window_results), default=0.0)
        matched = next((r.matched_seller_id for r in window_results if r.matched_seller_id), None)
        evaluations.append(
            WindowEvaluation(
                session_id=session.session_id,
                scenario=session.scenario,
                start_ms=label.start_ms,
                end_ms=label.end_ms,
                ground_truth=label.ground_truth,
                predicted=predicted,
                confidence=confidence,
                matched_seller_id=matched,
            ),
        )
    return evaluations


def evaluate_corpus(
    sessions: list[CorpusSession],
    *,
    config: AcousticFingerprintConfig | None = None,
) -> AcousticEvaluationReport:
    details: list[WindowEvaluation] = []
    for session in sessions:
        details.extend(evaluate_session(session, config=config))

    seller_total = sum(1 for item in details if item.ground_truth == 'seller')
    customer_total = sum(1 for item in details if item.ground_truth == 'customer')
    seller_hits = sum(
        1 for item in details if item.ground_truth == 'seller' and item.predicted == 'seller'
    )
    customer_false_seller = sum(
        1 for item in details if item.ground_truth == 'customer' and item.predicted == 'seller'
    )
    seller_false_customer = sum(
        1 for item in details if item.ground_truth == 'seller' and item.predicted == 'customer'
    )
    unknown_count = sum(1 for item in details if item.predicted == 'unknown')
    routing_hits = sum(
        1
        for item in details
        if item.ground_truth in {'seller', 'customer'}
        and item.predicted == item.ground_truth
    )
    routable = sum(1 for item in details if item.ground_truth in {'seller', 'customer'})

    seller_tpr = seller_hits / seller_total if seller_total else 0.0
    customer_fpr = customer_false_seller / customer_total if customer_total else 0.0
    seller_fnr = seller_false_customer / seller_total if seller_total else 0.0
    unknown_rate = unknown_count / len(details) if details else 0.0
    routing_accuracy = routing_hits / routable if routable else 0.0

    failures: list[str] = []
    if seller_tpr < 0.90:
        failures.append(f'seller_tpr below gate: {seller_tpr:.3f}')
    if customer_fpr > 0.02:
        failures.append(f'customer_false_seller_fpr above gate: {customer_fpr:.3f}')
    if seller_fnr > 0.05:
        failures.append(f'seller_false_customer_rate above gate: {seller_fnr:.3f}')
    if routing_accuracy < 0.95:
        failures.append(f'routing_accuracy below gate: {routing_accuracy:.3f}')

    return AcousticEvaluationReport(
        sessions=len(sessions),
        windows=len(details),
        seller_tpr=seller_tpr,
        customer_false_seller_fpr=customer_fpr,
        seller_false_customer_rate=seller_fnr,
        unknown_rate=unknown_rate,
        routing_accuracy=routing_accuracy,
        passed=not failures,
        failures=failures,
        details=details,
    )
