"""Rank candidates on one recorded training split using the task's own metric.

With --reuse-workers, read the existing headroom/*.joblib artifacts without fitting.
The legacy training mode measures a small expanded catalogue in a fresh directory.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(REPO))

import joblib
import numpy as np

from superagent.ab_comparison import macro_f1_pair_spread
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec, score_of
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


def rankings(rows):
    """Only comparable, actually measured candidates enter the two rankings."""
    if not rows:
        raise ValueError('No measured candidates on this partition')
    result = {}
    for key in ('search_score', 'search_accuracy'):
        order = sorted(rows, key=lambda name: (-rows[name][key], name))
        best, worst = rows[order[0]][key], rows[order[-1]][key]
        result[key] = dict(order=order, tied_best=[n for n in order if rows[n][key] == best],
                           best_score=best, worst_score=worst, spread_points=100 * (best - worst))
    return result


def paired(metric, labels, left, right, groups):
    y, a, b = map(np.asarray, (labels, left, right))
    # Scores that need probabilities cannot be reconstructed from predicted labels.
    result = dict(rows=len(y), groups=len(set(groups)), both_correct=int(np.sum((a == y) & (b == y))),
                  both_wrong=int(np.sum((a != y) & (b != y))), newly_correct=int(np.sum((a != y) & (b == y))),
                  newly_wrong=int(np.sum((a == y) & (b != y))), disagree=int(np.sum(a != b)))
    if not metric.requires_scores:
        result['right_minus_left_points'] = 100 * (metric.evaluate(y, b)['score'] - metric.evaluate(y, a)['score'])
    if metric.name == 'macro_f1':
        result['paired_bootstrap_sd_points'] = 100 * macro_f1_pair_spread(y, b, a, metric.labels, groups)
        result['group_bootstrap_draws'] = 128
    return result


def observed_coverage(metric, labels, predictions, selected):
    """Uses labels post hoc to count correctable rows among the observed candidates."""
    y = np.asarray(labels)
    guessed = np.asarray(predictions[selected])
    any_right = np.any(np.asarray(list(predictions.values())) == y, axis=0)
    result = dict(rows=len(y), selected_wrong=int(np.sum(guessed != y)), all_observed_wrong=int(np.sum(~any_right)),
                  correctable_among_observed=int(np.sum(any_right & (guessed != y))))
    if len(metric.labels) == 2 and not metric.requires_scores:
        optimistic = np.where(any_right, y, guessed)
        result['label_aware_oracle_primary'] = metric.evaluate(y, optimistic)['score']
    return result


def measure_worker(artifact, payload, payload_hash, candidate):
    metric = MetricSpec.from_payload(payload)
    labels = payload['y_search']
    record, facts = artifact['prediction_record'], artifact['metrics']
    config = facts['resolved_configuration']
    if (config['id'] != candidate or facts['primary_metric'] != metric.name
            or metric.check_identity(facts['metric_spec'])['status'] != 'confirmed'):
        raise ValueError('Candidate or measured metric is different from the expected task configuration')
    pred = record['predictions']
    if (record['evaluation_payload_sha256'] != payload_hash or record['label_sha256'] != fingerprint(labels)
            or record['prediction_sha256'] != fingerprint(pred) or record['rows'] != len(labels)):
        raise ValueError('Stored predictions do not bind to this training split')
    actual = metric.evaluate(labels, pred, artifact.get('search_scores'), artifact.get('score_classes'))
    if (score_of(facts) is None or not math.isclose(score_of(facts), actual['score'], abs_tol=1e-10, rel_tol=0)
            or not math.isclose(facts['search_accuracy'], actual['accuracy'], abs_tol=1e-10, rel_tol=0)):
        raise ValueError('Stored scores disagree with a fresh calculation from the recorded predictions')
    return dict(search_score=actual['score'], search_accuracy=actual['accuracy'],
                classifier_fits=facts.get('classifier_fits'), training_seconds=facts.get('training_seconds')), pred


def legacy_probe(run, timeout, reuse):
    from superagent.models import search_candidates
    from superagent.targeted_feedback import candidate_facts
    run = run.resolve()
    payload_path = run / 'inner_data.joblib'
    payload_hash = sha256(payload_path.read_bytes()).hexdigest()
    payload = joblib.load(payload_path)
    metric = MetricSpec.from_payload(payload)
    kind = payload['kind']
    task = TrainingTask(kind, payload['X_search'], np.asarray(payload['y_search']))
    groups = sample_groups(task, 'shared_text')
    menu = search_candidates(kind, 'expanded')
    directory = run / 'headroom'
    if not reuse:
        if len(menu) > 11:
            raise ValueError('Measuring this complete catalogue would exceed the eleven-fit allowance')
        directory.mkdir(exist_ok=False)
    rows, predictions, missing, failures = {}, {}, [], []
    start = time.monotonic()
    for index, candidate in enumerate(menu):
        name = candidate.family
        path = directory / f'{name}.joblib'
        if reuse:
            if not path.is_file():
                missing.append(name)
                continue
        else:
            from superagent.benchmark import train_candidate
            admission, ledger = path.with_suffix('.admission.json'), path.with_suffix('.fits.json')
            admission.write_text(json.dumps(dict(maximum_classifier_fits=1, ledger_path=str(ledger.resolve()))), encoding='utf-8')
            _, receipt = train_candidate(payload_path, name, path, timeout, fit_admission_path=admission)
            if receipt.get('status') != 'completed':
                failures.append(dict(candidate=name, status=receipt.get('status'), reason=receipt.get('error'),
                                     worker_wall_seconds=receipt.get('worker_wall_seconds'),
                                     physical_ledger=json.loads(ledger.read_text()) if ledger.exists() else None))
                missing.extend(c.family for c in menu[index + 1:])
                break
        row, predictions[name] = measure_worker(joblib.load(path), payload, payload_hash, name)
        rows[name] = dict(row, facts=candidate_facts(kind, {'id': name, 'params': candidate.params}))
    result = dict(kind=kind, primary_metric=metric.name, search_rows=len(task.y_train),
                  candidates=len(menu), evaluated=len(rows), missing=missing, failures=failures,
                  elapsed_seconds=round(time.monotonic() - start, 2),
                  mode='existing_worker_artifacts' if reuse else 'new_candidate_fits')
    if rows:
        rank = rankings(rows)
        best, worst = rank['search_score']['order'][0], rank['search_score']['order'][-1]
        result.update(best=best, worst=worst, best_score=rows[best]['search_score'], worst_score=rows[worst]['search_score'],
                      spread_points=rank['search_score']['spread_points'], ranks=rank,
                      best_vs_worst=paired(metric, task.y_train, predictions[worst], predictions[best], groups),
                      rows=[dict(candidate=n, **rows[n]) for n in rank['search_score']['order']])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--reuse-workers', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = legacy_probe(args.run_dir, args.timeout, args.reuse_workers)
    report = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(report)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report + '\n')
    if not result['evaluated'] or result['failures']:
        raise SystemExit(1)
