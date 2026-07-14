#!/usr/bin/env python3
"""Run the Phase 0 acoustic feasibility harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.acoustic_fingerprint.metrics import evaluate_corpus
from src.modules.acoustic_fingerprint.pcm_io import load_corpus_session
from src.modules.acoustic_fingerprint.synthetic_corpus import default_synthetic_sessions


def _load_sessions(corpus_dir: Path | None) -> list:
    if corpus_dir is None:
        return default_synthetic_sessions()
    sessions = []
    for child in sorted(corpus_dir.iterdir()):
        if child.is_dir() and (child / 'manifest.json').exists():
            sessions.append(load_corpus_session(child))
    if not sessions:
        raise SystemExit(f'No corpus sessions found under {corpus_dir}')
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description='Acoustic feasibility harness (Phase 0)')
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        help='Directory with recorded sessions (mic.wav, loopback.wav, manifest.json)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional JSON report output path',
    )
    args = parser.parse_args()

    sessions = _load_sessions(args.corpus_dir)
    report = evaluate_corpus(sessions)

    payload = {
        'sessions': report.sessions,
        'windows': report.windows,
        'seller_tpr': round(report.seller_tpr, 4),
        'customer_false_seller_fpr': round(report.customer_false_seller_fpr, 4),
        'seller_false_customer_rate': round(report.seller_false_customer_rate, 4),
        'unknown_rate': round(report.unknown_rate, 4),
        'routing_accuracy': round(report.routing_accuracy, 4),
        'passed': report.passed,
        'failures': report.failures,
        'details': [
            {
                'session_id': item.session_id,
                'scenario': item.scenario,
                'start_ms': item.start_ms,
                'end_ms': item.end_ms,
                'ground_truth': item.ground_truth,
                'predicted': item.predicted,
                'confidence': round(item.confidence, 4),
                'matched_seller_id': item.matched_seller_id,
            }
            for item in report.details
        ],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    return 0 if report.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
