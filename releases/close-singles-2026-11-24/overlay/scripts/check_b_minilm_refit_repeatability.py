"""Compare repeated MiniLM classifier fitting in independent processes without reading evaluation labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression

from scripts.run_b_autonomous_close_singles import ROOT, env, read, require, seal
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities
from superagent.data import load_development_task, load_prediction_features


PLAN = ROOT / 'docs/batches/B_minilm_refit_repeatability_20261124_zh.md'
TOLERANCES = {'default': 1e-4, 'tight': 1e-6}


def digest(*arrays):
    acc = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        acc.update(str((value.shape, value.dtype.str)).encode())
        acc.update(value.view(np.uint8))
    return acc.hexdigest()


def worker(a, index):
    trial = a.output / str(index)
    trial.mkdir(exist_ok=False)
    state = read(a.output / 'protocol.json')
    d = a.run / 'control/agent'
    require(ab_run.sha(d/'selected_model.joblib') == state['frozen_model'] and
            ab_run.sha(a.run/'anonymous.zip') == state['training'] and ab_run.sha(a.run/'features.zip') == state['features'],
            'Input differs from the sealed study')
    task = load_development_task(a.run/'anonymous.zip')
    require(len(task.y_train) == 7364 and task.y_val is None, 'Expected training-only input')
    rows = load_prediction_features(a.run/'features.zip','pair_text',read(d/'task_spec.json'))
    require(len(rows) == 937, 'Expected the held features only')
    first, second = joblib.load(d/'selected_model.joblib')['model'].models
    base = second.named_steps['classifier']
    require(base.tol == TOLERANCES['default'] and base.C == .3 and base.random_state == 42 and base.solver == 'lbfgs',
            'Expected the registered MiniLM head')
    features = second.named_steps['features']
    require(features.encoder_model == 'sentence-transformers/all-MiniLM-L6-v2', 'Different encoder')
    raw = features.transform(task.X_train)
    x = vstack([raw, -raw], format='csr')
    y = np.r_[task.y_train, 1-task.y_train]
    ordered = [(min(l,r), max(l,r)) for l,r in rows]
    direct, equal = np.array([l < r for l,r in rows]), np.array([l == r for l,r in rows])
    held = features.transform(ordered)
    e5 = canonical_probabilities(first,rows)[1][:,1]
    answer, data = {}, {}
    for label, tol in TOLERANCES.items():
        params = base.get_params().copy()
        params['tol'] = tol
        model = LogisticRegression(**params)
        with (trial/'physical.jsonl').open('a') as meter:
            meter.write(json.dumps(dict(classifier='LogisticRegression',tolerance=tol))+'\n')
        start = time.monotonic()
        model.fit(x,y)
        duration = time.monotonic()-start
        p = model.predict_proba(held)[:,1]
        oriented = np.where(equal,.5,np.where(direct,p,1-p))
        ensemble = (oriented+e5)/2
        hard = ((ensemble>.5)|((ensemble==.5)&direct)).astype(int)
        data.update({label+'_coef': model.coef_, label+'_intercept': model.intercept_,
                     label+'_probability': ensemble, label+'_hard': hard})
        answer[label] = dict(iterations=int(model.n_iter_[0]), fit_seconds=duration, features=int(x.shape[1]),
                             coef_sha256=digest(model.coef_,model.intercept_), probability_sha256=digest(ensemble),
                             predictions_sha256=digest(hard))
    with (trial/'weights_and_predictions.npz').open('xb') as f:
        np.savez_compressed(f,**data)
    seal(trial/'worker.json',dict(role=index, training_matrix=digest(x.data,x.indices,x.indptr), y=digest(y),
         held_matrix=digest(held.data,held.indices,held.indptr), e5_probability=digest(e5), observed=answer,
         output_sha256=ab_run.sha(trial/'weights_and_predictions.npz'), physical_sha256=ab_run.sha(trial/'physical.jsonl')))


def run(a):
    if a.worker is not None:
        worker(a, a.worker)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = read(a.run/'models_frozen.json')['arms']['control']
    seal(a.output/'protocol.json',dict(plan=ab_run.sha(PLAN), runner=ab_run.sha(Path(__file__)), source=ab_run.sources(),
         training=ab_run.sha(a.run/'anonymous.zip'), features=ab_run.sha(a.run/'features.zip'), frozen_model=frozen['model'],
         cold_study=ab_run.sha(a.run/'protocol.json'), tolerances=TOLERANCES, max_new_fits=4, model_api_cap=0, held_labels_allowed=False))
    summaries, runs = [], []
    for index in (0,1):
        flags = ['--worker',str(index),'--run',str(a.run),'--weights',str(a.weights),'--output',str(a.output)]
        with (a.output/f'worker_{index}.log').open('x') as log:
            subprocess.run([sys.executable,'-m','scripts.check_b_minilm_refit_repeatability',*flags], cwd=ROOT,
                           env=env(ROOT,a.weights,a.output/f'unexpected_native_worker_{index}.jsonl'), check=True,
                           stdout=log,stderr=subprocess.STDOUT,timeout=450)
        p = a.output / str(index)
        summary = read(p/'worker.json')
        require(ab_run.sha(p/'weights_and_predictions.npz') == summary['output_sha256'] and
                ab_run.sha(p/'physical.jsonl') == summary['physical_sha256'] and
                [json.loads(s)['tolerance'] for s in (p/'physical.jsonl').read_text().splitlines()] == list(TOLERANCES.values()),
                'Worker accounting differs')
        with np.load(p/'weights_and_predictions.npz',allow_pickle=False) as data:
            runs.append(dict(data))
        summaries.append(summary)
    def compare(a, b, left, right):
        return dict(coef_max_abs=float(np.max(np.abs(a[left+'_coef']-b[right+'_coef']))),
                    intercept_max_abs=float(np.max(np.abs(a[left+'_intercept']-b[right+'_intercept']))),
                    vote_probability_max_abs=float(np.max(np.abs(a[left+'_probability']-b[right+'_probability']))),
                    vote_label_differences=int(np.count_nonzero(a[left+'_hard'] != b[right+'_hard'])))
    repeat = {k:compare(runs[0],runs[1],k,k) for k in TOLERANCES}
    within = [compare(r,r,'default','tight') for r in runs]
    seal(a.output/'results.json',dict(protocol=ab_run.sha(a.output/'protocol.json'),
         worker_sha256=[ab_run.sha(a.output/str(i)/'worker.json') for i in (0,1)],
         paired_fingerprints_match={k:summaries[0][k]==summaries[1][k] for k in ('training_matrix','y','held_matrix','e5_probability')},
         iterations={k:[s['observed'][k]['iterations'] for s in summaries] for k in TOLERANCES},
         between_processes=repeat, within_process_default_vs_tight=within, training_rows=7364, held_features=937,
         physical_fits=4, model_api_calls=0, held_labels_read=False, accuracy_measured=False))
    seal(a.output/'completion.json',{k:ab_run.sha(a.output/(k+'.json')) for k in ('protocol','results')})
    print(json.dumps(read(a.output/'results.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=(0,1), type=int)
    for opt in ('run','weights','output'):
        parser.add_argument('--'+opt, required=True, type=lambda s:Path(s).resolve())
    run(parser.parse_args())
