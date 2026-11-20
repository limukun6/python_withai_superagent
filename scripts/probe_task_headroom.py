"""Measure how much candidate choice can matter on one task.

Fits every candidate in the expanded portfolio on the same training-internal
split a recorded run already created, then reports the spread and the binomial
standard error of each accuracy. A task whose candidate spread is inside one
standard error cannot demonstrate that selection helps, whatever the agent does.

    python scripts/probe_task_headroom.py --run-dir <run with inner_data.joblib>
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / 'runs' / 'integration'
if not (INTEGRATION / 'superagent').is_dir():
    INTEGRATION = REPO
sys.path.insert(0, str(INTEGRATION))

import joblib

from superagent.benchmark import train_candidate
from superagent.models import search_candidates
from superagent.targeted_feedback import candidate_facts


def standard_error(p, n):
    return math.sqrt(p * (1 - p) / n) if 0 < p < 1 else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--output')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    payload_path = run / 'inner_data.joblib'
    payload = joblib.load(payload_path)
    kind = payload['kind']
    n_search = len(payload['y_search'])
    rows = []
    started = time.monotonic()
    for candidate in search_candidates(kind, 'expanded'):
        name = candidate.family
        destination = run / 'headroom' / f'{name}.joblib'
        destination.parent.mkdir(parents=True, exist_ok=True)
        _, metrics = train_candidate(payload_path, name, destination, args.timeout)
        score = metrics.get('search_accuracy')
        rows.append({'candidate': name, 'status': metrics.get('status'),
                     'search_accuracy': score,
                     'standard_error': standard_error(score, n_search) if score is not None else None,
                     'training_seconds': round(metrics.get('training_seconds', 0), 1),
                     'classifier_fits': metrics.get('classifier_fits'),
                     'facts': candidate_facts(kind, {'id': name, 'params': candidate.params})})
        print(f'{name}: {score} ({rows[-1]["training_seconds"]}s)', flush=True)
    scored = [r for r in rows if r['search_accuracy'] is not None]
    scored.sort(key=lambda r: -r['search_accuracy'])
    best, worst = scored[0], scored[-1]
    se = best['standard_error']
    summary = {
        'run_dir': str(run), 'kind': kind, 'candidates': len(rows),
        'evaluated': len(scored), 'search_rows': n_search,
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'best': best['candidate'], 'best_score': round(best['search_accuracy'], 6),
        'worst': worst['candidate'], 'worst_score': round(worst['search_accuracy'], 6),
        'spread_points': round(100 * (best['search_accuracy'] - worst['search_accuracy']), 2),
        'standard_error_points': round(100 * se, 2),
        'spread_in_standard_errors': round((best['search_accuracy'] - worst['search_accuracy']) / se, 2),
        'candidates_within_one_se_of_best': sum(
            1 for r in scored if best['search_accuracy'] - r['search_accuracy'] <= se),
        'rows': sorted(rows, key=lambda r: -(r['search_accuracy'] or 0)),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
