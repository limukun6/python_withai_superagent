"""Measure a fixed numeric kernel against its exact linear control on training-only groups."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np
import sklearn
from sklearn.base import clone

from scripts import b_numeric_kernel, run_b_numeric_geometry as geometry
from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from superagent.ab_run import sha
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_numeric_kernel_1123_zh.md'
SALT = 'numeric-kernel-confirm-1123-v1'


def disjoint_panels(groups, prior_indices):
    examined = set(groups[np.unique(np.concatenate(prior_indices))])
    buckets = {}
    for key in set(groups) - examined:
        digest = hashlib.sha256(f'{SALT}:{key}'.encode()).digest()
        buckets[key] = int.from_bytes(digest, 'big') % 5
    assignment = np.array([buckets.get(key, -1) for key in groups])
    panels = [np.flatnonzero(assignment == bucket) for bucket in (0, 1)]
    train = np.flatnonzero((assignment != 0) & (assignment != 1))
    first, second = [set(groups[ids]) for ids in panels]
    check(first and second and first.isdisjoint(second)
          and examined.isdisjoint(first | second)
          and set(groups[train]).isdisjoint(first | second), 'Invalid confirmation panels')
    return train, panels


def fit_once(folder, spec, x, y, train):
    folder.mkdir(exist_ok=True)
    started = folder / 'fit_started.json'
    finished = folder / 'fit_completed.json'
    weights = folder / 'classifier.joblib'
    record = dict(method=folder.name, training_order=fingerprint(train),
                  training_labels=fingerprint(y[train]), rows=len(train), attempts=1)
    if not finished.exists():
        check(not weights.exists(), 'A previous fit left weights without a receipt')
        save(started, record)
        begin = time.monotonic()
        model = spec.fit(x[train], y[train])
        elapsed = time.monotonic() - begin
        with weights.open('xb') as stream:
            joblib.dump(model, stream)
        save(finished, dict(fits=1, seconds=elapsed, weights=sha(weights), started=sha(started)))
    receipt = read(finished)
    check(read(started) == record and receipt['weights'] == sha(weights)
          and receipt['started'] == sha(started), 'Fit or training inputs changed')
    return joblib.load(weights)


def predictions(model, x, classes):
    prob = model.predict_proba(x)
    labels = model.predict(x)
    np.testing.assert_array_equal(model.classes_, classes)
    check(np.isfinite(prob).all() and (prob >= 0).all()
          and np.allclose(prob.sum(axis=1), 1.0), 'Invalid class probabilities')
    np.testing.assert_array_equal(classes[prob.argmax(axis=1)], labels)
    return labels


def seal_predictions(folder, ids, baseline, kernel):
    path = folder / 'predictions.npz'
    expected = dict(row_ids=ids, baseline=baseline, kernel=kernel)
    if not path.exists():
        with path.open('xb') as stream:
            np.savez_compressed(stream, **expected)
    with np.load(path, allow_pickle=False) as stored:
        for key, values in expected.items():
            np.testing.assert_array_equal(stored[key], values)
    frozen_save(folder / 'exports.json', dict(file=sha(path), rows=len(ids),
                fingerprints={key: fingerprint(values) for key, values in expected.items()}))


def score(folder, y, groups):
    check(sha(folder / 'predictions.npz') == read(folder / 'exports.json')['file'], 'Export changed')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
        ids, base, new = [data[k] for k in ('row_ids', 'baseline', 'kernel')]
    truth, keys = y[ids], groups[ids]
    counts = {}
    for key, old_ok, new_ok in zip(keys, base == truth, new == truth, strict=True):
        counts.setdefault(key, []).append(int(old_ok) + 2 * int(new_ok))
    frozen_save(folder / 'grouped_correctness.json', [sorted(counts[k]) for k in sorted(counts)])
    return geometry.paired_scores(base, new, truth, keys)


def run(control, out):
    sys.addaudithook(offline)
    references = geometry.reference(control)
    full = control / 'refit_data.joblib'
    check(sha(full) == 'c2c09a12460cbd5291855d10c5d04607dad6c7b595c7d6820835832e1edf8398',
          'Unrecognized original training pool')
    raw = joblib.load(full)
    x, y = raw['X_train'], np.asarray(raw['y_train'])
    groups = sample_groups(SimpleNamespace(kind='numeric_tabular', X_train=x))
    splits = [read(control / (part + '_split.json')) for part in geometry.PARTS]
    train, screen = [np.asarray(splits[0][k]) for k in ('train_indices', 'search_indices')]
    check(set(groups[train]).isdisjoint(groups[screen]), 'Duplicate group in historical screen')
    confirmation_train, panels = disjoint_panels(groups, [s['search_indices'] for s in splits])
    worker = joblib.load(control / references['primary']['trial'])
    classes = np.unique(y[train])
    primary_base = predictions(worker['model'], x[screen], classes)
    np.testing.assert_array_equal(primary_base, worker['search_predictions'])
    check(fingerprint(primary_base) == worker['prediction_record']['prediction_sha256']
          and fingerprint(y[screen]) == references['primary']['label']
          and fingerprint(screen) == splits[0]['row_ids_sha256'], 'Archived control differs')
    out.mkdir(parents=True, exist_ok=True)
    partitions = dict(screen_fit=train, screen_eval=screen, confirm_fit=confirmation_train,
                      confirm_a=panels[0], confirm_b=panels[1])
    frozen_save(out / 'protocol.json', dict(plan=sha(PLAN), runner=sha(Path(__file__)),
        features=sha(Path(b_numeric_kernel.__file__)), reference_helper=sha(Path(geometry.__file__)),
        training=sha(full), prior=references, sklearn=sklearn.__version__, salt=SALT,
        partitions={k: dict(rows=len(v), order=fingerprint(v), groups=len(set(groups[v])))
                    for k, v in partitions.items()}, fit_cap=3, model_api_cap=0))
    primary = out / 'primary'
    primary.mkdir(exist_ok=True)
    candidate = fit_once(primary / 'kernel', b_numeric_kernel.add_kernel(worker['model']), x, y, train)
    seal_predictions(primary, screen, primary_base, predictions(candidate, x[screen], classes))
    results = {'primary': score(primary, y, groups)}
    qualified = results['primary']['new'] - results['primary']['baseline'] >= 4
    frozen_save(out / 'screen_decision.json', dict(continue_confirmation=qualified,
                screen=sha(primary / 'exports.json'), threshold_rows=4))
    if qualified:
        refit = out / 'confirm_models'
        refit.mkdir(exist_ok=True)
        specs = [clone(worker['model']), b_numeric_kernel.add_kernel(worker['model'])]
        baseline, kernel = [fit_once(refit / name, spec, x, y, confirmation_train)
                            for name, spec in zip(('baseline', 'kernel'), specs, strict=True)]
        for name, ids in zip(('confirm_a', 'confirm_b'), panels, strict=True):
            folder = out / name
            folder.mkdir(exist_ok=True)
            seal_predictions(folder, ids, predictions(baseline, x[ids], classes),
                             predictions(kernel, x[ids], classes))
        for name in ('confirm_a', 'confirm_b'):
            results[name] = score(out / name, y, groups)
    deltas = [v['new'] - v['baseline'] for k, v in results.items() if k != 'primary']
    fit_receipts = {str(p.relative_to(out)): sha(p) for p in sorted(out.glob('*/*/fit_completed.json'))}
    check(len(fit_receipts) == (3 if qualified else 1), 'Wrong number of physical fits')
    eligible = len(deltas) == 2 and min(deltas) >= 0 and sum(deltas) >= 8
    frozen_save(out / 'results.json', dict(panels=results, eligible_for_outer=eligible,
                physical_fits=len(fit_receipts), model_api_calls=0, new_rmb=0, outer_opened=False))
    frozen_save(out / 'completion.json', dict(protocol=sha(out / 'protocol.json'),
                results=sha(out / 'results.json'), decision=sha(out / 'screen_decision.json'),
                exports={k: sha(out / k / 'exports.json') for k in results},
                groups={k: sha(out / k / 'grouped_correctness.json') for k in results},
                fit_receipts=fit_receipts))
    print(json.dumps(dict(panels={k: [v['baseline'], v['new'], v['rows']] for k, v in results.items()},
                          eligible=eligible, fits=len(fit_receipts))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('control', 'out'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run(args.control.resolve(), args.out.resolve())
