"""Probe Task1 candidates that exist but are excluded from the ab-v1 portfolio.

Reuses the exact split payloads from the headroom screening, so the comparison
against catboost_depth8_reg3.0 is same-rows and same-metric. No API.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model

SPLITS = REPO / 'research' / 'task1_headroom' / 'attempt_02'
TASK = r'D:\codex\python_withai\data\released_task1.zip'
BASELINE = {707: 0.753391, 811: 0.741677, 823: 0.742910}
CASES = (
    ('category_cross_C0.1', (707, 811, 823), 1),
    ('adaptive_catboost_depth8_reg3.0', (707,), 2),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=REPO / 'research' / 'task1_extra_candidates' / 'attempt_01')
    parser.add_argument('--timeout', type=float, default=1800.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    planned = sum(len(seeds) * fits for _, seeds, fits in CASES)
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', 'planned_fits': planned}))
        return 0
    task = load_development_task(TASK)
    metric = MetricSpec.for_task(task)
    rows = []
    for candidate, seeds, expected in CASES:
        for seed in seeds:
            payload = SPLITS / f'split_{seed}.joblib'
            destination = out / f'trial_{seed}_{candidate}.joblib'
            admission = destination.with_suffix('.admission.json')
            admission.write_text(json.dumps({'maximum_classifier_fits': expected,
                                             'ledger_path': str(destination.with_suffix('.fits.json'))}),
                                 encoding='utf-8')
            started = time.monotonic()
            model, facts = train_candidate(payload, candidate, destination, args.timeout,
                                           fit_admission_path=admission)
            elapsed = round(time.monotonic() - started, 3)
            if model is None:
                rows.append({'candidate': candidate, 'seed': seed, 'status': facts['status'],
                             'error': str(facts.get('error'))[:300], 'seconds': elapsed})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
                continue
            data = joblib.load(payload)
            predictions = np.asarray(predict_with_model(task.kind, model, data['X_search'])).reshape(-1)
            accuracy = float((predictions == np.asarray(data['y_search'])).mean())
            record = {'candidate': candidate, 'seed': seed, 'status': facts['status'],
                      'search_accuracy': round(accuracy, 6),
                      'baseline_depth8': BASELINE[seed],
                      'delta_pp': round(100 * (accuracy - BASELINE[seed]), 3),
                      'seconds': elapsed,
                      'fits': json.loads(destination.with_suffix('.fits.json').read_text(encoding='utf-8'))}
            rows.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'fits'}, ensure_ascii=False), flush=True)
    (out / 'results.json').write_text(json.dumps({'rows': rows}, indent=2, ensure_ascii=False) + '\n',
                                      encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
