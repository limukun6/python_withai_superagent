"""Frozen migration evaluation: two candidates on three internal splits, then one delivery score.

Pre-registered in docs/batches/TASK1_MIGRATION_PLAN_20261125_zh.md.
Six fits decide the promotion; two more refit each candidate on the full training
split for the designated-validation report. No API, no selection from validation.
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
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model
from superagent.splits import grouped_partition

SEEDS = (707, 811, 823)
CANDIDATES = (('baseline', 'catboost_depth8_reg3.0'),
              ('interactions', 'supported_field_interactions'))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(payload, candidate, destination, timeout):
    admission = destination.with_suffix('.admission.json')
    admission.write_text(json.dumps({'maximum_classifier_fits': 1,
                                     'ledger_path': str(destination.with_suffix('.fits.json'))}),
                         encoding='utf-8')
    started = time.monotonic()
    model, facts = train_candidate(payload, candidate, destination, timeout,
                                   fit_admission_path=admission)
    return model, facts, round(time.monotonic() - started, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=str(REPO / 'research' / 'task1_migration' /
                                              'car_evaluation_categorical.zip'))
    parser.add_argument('--output', type=Path, default=REPO / 'research' / 'task1_migration' / 'attempt_01')
    parser.add_argument('--timeout', type=float, default=900.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    task = load_development_task(args.task)
    metric = MetricSpec.for_task(task)
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'batch': 'task1-migration', 'task_sha256': sha(args.task),
                'task_rows': int(len(task.y_train)), 'kind': task.kind,
                'metric': metric.record(), 'seeds': list(SEEDS),
                'candidates': [{'role': role, 'id': candidate} for role, candidate in CANDIDATES],
                'planned_fits': 6, 'delivery_fits': 2, 'api': 0,
                'validation_used_for_selection': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n',
                                       encoding='utf-8')
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', **{k: protocol[k] for k in
                                                                  ('planned_fits', 'delivery_fits', 'kind')}}))
        return 0

    rows = []
    for seed in SEEDS:
        train, held, resource = grouped_partition(task, seed, 'exact_rows')
        payload = out / f'split_{seed}.joblib'
        joblib.dump(dict(kind=task.kind, X_train=task.X_train[train], y_train=task.y_train[train],
                         X_search=task.X_train[held], y_search=task.y_train[held],
                         metric_spec=metric.record()), payload)
        for role, candidate in CANDIDATES:
            model, facts, seconds = fit(payload, candidate, out / f'trial_{seed}_{candidate}.joblib',
                                        args.timeout)
            data = joblib.load(payload)
            predictions = np.asarray(predict_with_model(task.kind, model, data['X_search'])).reshape(-1)
            rows.append({'seed': seed, 'role': role, 'candidate': candidate,
                         'search_score': round(float(facts['search_metrics']['score']), 6),
                         'seconds': seconds, 'split': resource,
                         'predictions': [str(v) for v in predictions],
                         'labels': [str(v) for v in data['y_search']]})
            print(json.dumps({k: v for k, v in rows[-1].items() if k not in ('predictions', 'labels', 'split')},
                             ensure_ascii=False), flush=True)

    comparisons = []
    for seed in SEEDS:
        group = {row['candidate']: row for row in rows if row['seed'] == seed}
        base, other = group['catboost_depth8_reg3.0'], group['supported_field_interactions']
        truth = np.asarray(base['labels'], dtype=object)
        base_ok = np.asarray(base['predictions'], dtype=object) == truth
        other_ok = np.asarray(other['predictions'], dtype=object) == truth
        comparisons.append({'seed': seed,
                            'delta_pp': round(float(100 * (other_ok.mean() - base_ok.mean())), 3),
                            'fixed': int((~base_ok & other_ok).sum()),
                            'broken': int((base_ok & ~other_ok).sum()),
                            'shared_errors': int((~base_ok & ~other_ok).sum())})
    values = [row['delta_pp'] for row in comparisons]
    promotion = {'mean_delta_pp': round(sum(values) / len(values), 3),
                 'positive_splits': int(sum(v > 0 for v in values)),
                 'worst_pp': min(values),
                 'passes': sum(values) / len(values) >= 0.5 and sum(v > 0 for v in values) >= 2
                           and min(values) >= -0.5}

    delivery = {}
    full = out / 'full_train.joblib'
    joblib.dump(dict(kind=task.kind, X_train=task.X_train, y_train=task.y_train,
                     X_search=task.X_val, y_search=task.y_val,
                     metric_spec=metric.record()), full)
    for role, candidate in CANDIDATES:
        model, facts, seconds = fit(full, candidate, out / f'delivery_{candidate}.joblib', args.timeout)
        predictions = np.asarray(predict_with_model(task.kind, model, task.X_val)).reshape(-1)
        delivery[candidate] = {'validation_score': round(float((predictions == np.asarray(task.y_val)).mean()), 6),
                               'seconds': seconds}
    summary = {'internal': comparisons, 'promotion': promotion, 'delivery': delivery,
               'fits': len(rows) + len(delivery)}
    (out / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2,
                                                 ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
