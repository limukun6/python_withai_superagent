"""Task1 headroom screening: three candidates on three training-internal splits.

Pre-registered in docs/batches/TASK1_HEADROOM_PLAN_20261125_zh.md. The runner
validates by default; `--execute` trains, and it refuses to exceed the nine
physical-fit cap. No API call, no designated-validation read, no formal test.
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


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    return value

from superagent.benchmark import train_candidate
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model
from superagent.splits import grouped_partition
from superagent.contracts import TaskSpec

SEEDS = (707, 811, 823)
CANDIDATES = (
    ('baseline', 'catboost_depth8_reg3.0'),
    ('regularized', 'catboost_depth6_reg10.0'),
    ('interactions', 'supported_field_interactions'),
)
FIT_CAP = 9
EVIDENCE = REPO / 'research' / 'task1_headroom'


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_payload(task, metric, seed, directory):
    train, held, resource = grouped_partition(task, seed, 'exact_rows')
    payload = directory / f'split_{seed}.joblib'
    contents = dict(kind=task.kind, X_train=task.X_train[train], y_train=task.y_train[train],
                    X_search=task.X_train[held], y_search=task.y_train[held],
                    metric_spec=metric.record(),
                    pair_semantics=getattr(task, 'spec', {}).get('pair_semantics'))
    joblib.dump(contents, payload)
    return payload, held, train, resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=r'D:\codex\python_withai\data\released_task1.zip')
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'attempt_01')
    parser.add_argument('--timeout', type=float, default=900.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    task = load_development_task(args.task)
    metric = MetricSpec.for_task(task)
    spec = TaskSpec.from_task(task).record()
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'batch': 'task1-headroom', 'seeds': list(SEEDS),
                'candidates': [{'role': role, 'id': candidate} for role, candidate in CANDIDATES],
                'fit_cap': FIT_CAP, 'metric': metric.record(), 'task_kind': spec['kind'],
                'train_rows': int(len(task.y_train)), 'task_sha256': sha(args.task),
                'source_files': {p.name: sha(p) for p in sorted((REPO / 'superagent').glob('*.py'))},
                'designated_validation_read': False, 'formal_test_read': False,
                'automatic_retries': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', 'splits': len(SEEDS),
                          'candidates': len(CANDIDATES), 'planned_fits': len(SEEDS) * len(CANDIDATES)}))
        return 0

    rows, fits = [], 0
    for seed in SEEDS:
        payload, held, train, resource = build_payload(task, metric, seed, out)
        split_record = {'seed': seed, 'train_rows': int(len(train)), 'search_rows': int(len(held)),
                        'groups': resource}
        for role, candidate in CANDIDATES:
            if fits >= FIT_CAP:
                raise RuntimeError(f'fit cap {FIT_CAP} reached; refusing to continue')
            destination = out / f'trial_{seed}_{candidate}.joblib'
            admission = destination.with_suffix('.admission.json')
            admission.write_text(json.dumps({'maximum_classifier_fits': 1,
                                             'ledger_path': str(destination.with_suffix('.fits.json'))}),
                                 encoding='utf-8')
            started = time.monotonic()
            model, facts = train_candidate(payload, candidate, destination, args.timeout,
                                           fit_admission_path=admission)
            elapsed = round(time.monotonic() - started, 3)
            if model is None:
                rows.append({'seed': seed, 'role': role, 'candidate': candidate, 'status': facts['status'],
                             'error': facts.get('error'), 'seconds': elapsed})
                continue
            data = joblib.load(payload)
            predictions = np.asarray(predict_with_model(task.kind, model, data['X_search'])).reshape(-1)
            fit_record = json.loads(destination.with_suffix('.fits.json').read_text(encoding='utf-8'))
            fits += fit_record.get('attempted_classifier_fits', 0)
            rows.append({'seed': seed, 'role': role, 'candidate': candidate, 'status': facts['status'],
                         'search_score': round(float(facts['search_metrics']['score']), 6),
                         'accuracy': round(float(facts['search_metrics'].get('accuracy', 0)), 6),
                         'seconds': elapsed, 'fits': fit_record.get('attempted_classifier_fits', 0),
                         'predictions': predictions.tolist(), 'labels': data['y_search'].tolist(),
                         'split': split_record})
            print(json.dumps({k: v for k, v in rows[-1].items() if k not in ('predictions', 'labels')},
                             ensure_ascii=False), flush=True)

    by_seed = {}
    for row in rows:
        by_seed.setdefault(row['seed'], {})[row['candidate']] = row
    comparisons = []
    for seed, group in sorted(by_seed.items()):
        baseline = group.get('catboost_depth8_reg3.0')
        if not baseline or 'predictions' not in baseline:
            continue
        truth = np.asarray(baseline['labels'])
        base_ok = np.asarray(baseline['predictions']) == truth
        for role, candidate in CANDIDATES:
            if role == 'baseline':
                continue
            other = group.get(candidate)
            if not other or 'predictions' not in other:
                continue
            other_ok = np.asarray(other['predictions']) == truth
            comparisons.append({'seed': seed, 'challenger': candidate,
                                'delta_pp': round(100 * (other_ok.mean() - base_ok.mean()), 3),
                                'fixed': int((~base_ok & other_ok).sum()),
                                'broken': int((base_ok & ~other_ok).sum()),
                                'shared_errors': int((~base_ok & ~other_ok).sum())})
    summary = {'runs': len(rows), 'fits': fits, 'comparisons': comparisons}
    for candidate in {c for _, c in CANDIDATES}:
        values = [c['delta_pp'] for c in comparisons if c['challenger'] == candidate]
        if not values:
            continue
        summary[candidate] = {'mean_delta_pp': round(sum(values) / len(values), 3),
                              'positive_splits': sum(v > 0 for v in values),
                              'worst_pp': min(values), 'splits': len(values)}
    (out / 'results.json').write_text(json.dumps(native({'summary': summary, 'rows': rows}), indent=2,
                                                 ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
