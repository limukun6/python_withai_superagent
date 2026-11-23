"""Remeasure two fixed E5 training folds with one group-weighted logistic head each."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.base import clone

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from superagent import ab_run
from superagent.ab_methods import TRANSFER_METHODS
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text, sample_groups

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_shared_title_weighting_1123_zh.md'
SEEDS = (18643, 40531)


def group_ids(rows):
    return sample_groups(SimpleNamespace(kind='pair_text', X_train=rows), 'shared_text')


def bind(folder, seed):
    manifest = read(folder/'run_manifest.json')
    check(manifest['seed'] == seed, 'Wrong archived seed')
    records = [r for r in read(folder/'history.json') if r['candidate'] == TRANSFER_METHODS['deep-e5'][1]
               and r['split_id'] == 'primary']
    check(len(records) == 1, 'Expected one original E5 primary measurement')
    r = records[0]
    trial = folder / ('trial_' + str(r['trial']) + '.joblib')
    payload = folder/'primary.joblib'
    check(ab_run.sha(payload) == r['evaluation_payload_sha256'], 'Archived training fold differs')
    return dict(payload=ab_run.sha(payload), trial=trial.name, trial_sha=ab_run.sha(trial),
                labels=r['label_sha256'], trial_score=r['search_score'])


def measure(folder, dest, ref):
    dest.mkdir(exist_ok=True)
    source = joblib.load(folder/'primary.joblib')
    previous = joblib.load(folder/ref['trial'])
    check(previous['prediction_record']['evaluation_payload_sha256'] == ref['payload'] and
          previous['prediction_record']['label_sha256'] == ref['labels'], 'Original worker record differs')
    features, original_head = previous['model'].steps[0][1], previous['model'].steps[1][1]
    check(features.variant == 'e5' and original_head.C == .3, 'Unexpected original features or head')
    train, search = source['X_train'], source['X_search']
    train_y = np.asarray(source['y_train'])
    training_titles = {normalized_text(s) for row in train for s in row}
    check(not any(normalized_text(s) in training_titles for row in search for s in row), 'Shared title at fold boundary')
    train_groups = group_ids(train)
    sizes = Counter(train_groups)
    weights = np.array([sizes[g] ** -.5 for g in train_groups])
    weights /= weights.mean()
    check(weights.min() < 1 < weights.max() and np.isclose(weights.mean(), 1), 'Degenerate group weights')
    search_x = features.transform(search)
    base = original_head.predict(search_x)
    np.testing.assert_array_equal(base, previous['search_predictions'])
    check(fingerprint(base) == previous['prediction_record']['prediction_sha256'], 'Original prediction hash mismatch')
    model_path, attempted = dest/'classifier.joblib', dest/'fit_started.json'
    if not model_path.exists():
        train_x = features.transform(train)
        save(attempted, dict(classifier=type(original_head).__name__, total_attempts=1,
                              unaugmented_rows=len(train), weight_min=float(weights.min()),
                              weight_max=float(weights.max())))
        weighted = clone(original_head).fit(vstack((train_x, -train_x)), np.r_[train_y, 1-train_y],
                                            sample_weight=np.tile(weights,2))
        with model_path.open('xb') as f:
            joblib.dump(weighted, f)
        save(dest/'fit_completed.json', dict(model=ab_run.sha(model_path), fit_record=ab_run.sha(attempted),
                                             new_supervised_fits=1))
    completed = read(dest/'fit_completed.json')
    check(completed['model'] == ab_run.sha(model_path) and completed['fit_record'] == ab_run.sha(attempted)
          and completed['new_supervised_fits'] == 1, 'Incomplete fit evidence; no automatic refit')
    new = joblib.load(model_path).predict(search_x)
    prediction_file = dest/'predictions.npz'
    if not prediction_file.exists():
        with prediction_file.open('xb') as f:
            np.savez_compressed(f, original=base, weighted=new)
    with np.load(prediction_file) as saved:
        np.testing.assert_array_equal(base, saved['original'])
        np.testing.assert_array_equal(new, saved['weighted'])
    frozen_save(dest/'exports.json', dict(original=fingerprint(base), weighted=fingerprint(new),
                                         file=ab_run.sha(prediction_file), model=completed['model']))
    return set(hashlib.sha256(json.dumps(sorted(map(normalized_text,row)),
               ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() for row in search)


def score(folder, dest, ref):
    source = joblib.load(folder/'primary.joblib')
    y = np.asarray(source['y_search'])
    check(fingerprint(y) == ref['labels'], 'Scoring labels differ')
    with np.load(dest/'predictions.npz') as rows:
        old = rows['original'] == y
        new = rows['weighted'] == y
    check(float(old.mean()) == ref['trial_score'], 'Cannot reconstruct original accuracy')
    delta = new.astype(int)-old.astype(int)
    groups = group_ids(source['X_search'])
    counts = Counter(group_ids(source['X_train']))
    return dict(training_rows=len(source['X_train']), training_groups=len(counts), max_component=max(counts.values()),
                rows=len(y), original=int(old.sum()), weighted=int(new.sum()), gained=int((delta>0).sum()),
                lost=int((delta<0).sum()), delta=paired(delta,groups), shared_train_search_titles=0)


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    folders = dict(zip(SEEDS, (a.first, a.second)))
    references = {str(s): bind(folder,s) for s,folder in folders.items()}
    frozen_save(a.out/'protocol.json', dict(plan=ab_run.sha(PLAN), runner=ab_run.sha(Path(__file__)),
                references=references, fits_cap=2, model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    overlap = [measure(folder, a.out/str(s), references[str(s)]) for s,folder in folders.items()]
    result = {str(s): score(folder,a.out/str(s),references[str(s)]) for s,folder in folders.items()}
    nets = [r['weighted']-r['original'] for r in result.values()]
    frozen_save(a.out/'results.json', dict(folds=result, search_pairs_observed_in_both=len(set.intersection(*overlap)),
            eligible_for_confirmation=min(nets)>=0 and max(nets)>=5, fits=2, model_api_calls=0, new_rmb=0,
            development_reserve_used=False, designated_validation_used=False))
    frozen_save(a.out/'completion.json', dict(results=ab_run.sha(a.out/'results.json'),
                protocol=ab_run.sha(a.out/'protocol.json'), physical_fits=sum(read(a.out/str(s)/'fit_completed.json')
                ['new_supervised_fits'] for s in SEEDS), model_api_calls=0))
    print(json.dumps(read(a.out/'results.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace','first','second','out'):
        p.add_argument('--'+key, type=lambda v: Path(v).resolve(), required=True)
    run(p.parse_args())
