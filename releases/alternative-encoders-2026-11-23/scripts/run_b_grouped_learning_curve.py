"""Measure a fixed preference ensemble as its training pool grows by whole title groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures, fit_sentence
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_grouped_learning_curve_1123_zh.md'
ORDERS = ('curve-a', 'curve-b')


def subsets(pool, groups):
    result = {}
    distinct = set(groups[pool])
    for salt in ORDERS:
        def rank(group):
            return hashlib.sha256((salt + ':' + group).encode('ascii')).digest(), group
        ordered = sorted(distinct, key=rank)
        for percent in (50, 75):
            allowed = set(ordered[:len(ordered) * percent // 100])
            result[f'{salt}-{percent}'] = pool[np.isin(groups[pool], list(allowed))]
        check(set(result[salt+'-50']) < set(result[salt+'-75']) < set(pool), 'Subsets must be nested')
    result['full'] = pool
    check(len({fingerprint(v) for v in result.values()}) == 5, 'Expected five distinct fitting sets')
    return result


def fit_member(folder, name, rows, y):
    folder.mkdir(exist_ok=True)
    model = folder / 'model.joblib'
    started = folder / 'fit_started.json'
    completed = folder / 'fit_completed.json'
    if not completed.exists():
        check(not started.exists() and not model.exists(), 'Unresolved fit in '+str(folder))
        save(started, dict(component=name, rows=len(rows), fits=1))
        t = time.monotonic()
        pipeline, _ = fit_sentence(rows, y, rows[:1], 'pair_text', alternative='e5' if name=='e5' else None,
                                   familiarity=name=='familiarity')
        with model.open('xb') as stream:
            joblib.dump(pipeline, stream)
        save(completed, dict(model=sha(model), start=sha(started), fits=1, seconds=time.monotonic()-t))
    receipt = read(completed)
    check(receipt['model'] == sha(model) and receipt['start'] == sha(started), 'Fitted component changed')
    return joblib.load(model)


def predict_all(out, sets, rows, labels, both, control):
    search = [rows[i] for i in both]
    exports = {}
    for key, indices in sets.items():
        folder = out / key
        folder.mkdir(exist_ok=True)
        train = [rows[i] for i in indices]
        if key == 'full':
            features = AlternativeSentenceFeatures('pair_text', 'e5').fit(train)
            e5 = Pipeline([('features', features), ('classifier', joblib.load(control / 'regular/classifier.joblib'))])
            np.testing.assert_array_equal(e5.predict(search), np.load(control / 'regular/predictions.npy', allow_pickle=False))
        else:
            e5 = fit_member(folder / 'e5', 'e5', train, labels[indices])
        fam = fit_member(folder / 'familiarity', 'familiarity', train, labels[indices])
        ensemble = PreferenceProbabilityVote('pair_text', [e5, fam], np.array([0,1]))
        pred = ensemble.predict(search)
        sample = search[:17]
        swapped = [(r,l) for l,r in sample]
        np.testing.assert_array_equal(ensemble.predict(swapped), 1-pred[:len(sample)])
        file = folder / 'predictions.npy'
        if not file.exists():
            with file.open('xb') as stream:
                np.save(stream, pred)
        np.testing.assert_array_equal(np.load(file, allow_pickle=False), pred)
        exports[key] = dict(file=sha(file), prediction=fingerprint(pred), training=fingerprint(indices),
                            fits={m:sha(folder / m / 'fit_completed.json') for m in
                                  (('familiarity',) if key=='full' else ('e5','familiarity'))})
    frozen_save(out / 'exports.json', exports)
    return exports


def summarize(out, sets, parts, groups, labels, both, exports):
    panels = {'A':slice(0,len(parts['A'])), 'B':slice(len(parts['A']),None), 'combined':slice(None)}
    hits, scores, transitions = {}, {}, {}
    for key in sets:
        file = out / key / 'predictions.npy'
        check(sha(file) == exports[key]['file'], 'Changed predictions')
        hits[key] = np.load(file, allow_pickle=False) == labels[both]
        scores[key] = {panel:int(hits[key][ii].sum()) for panel,ii in panels.items()}
    for salt in ORDERS:
        for low, high in [('50','75'),('75','100'),('50','100')]:
            old = hits[salt+'-'+low]
            new = hits['full' if high=='100' else salt+'-'+high]
            d = new.astype(int) - old.astype(int)
            transitions[salt+'-'+low+'-'+high] = {
                panel:dict(gained=int((d[ii]>0).sum()), lost=int((d[ii]<0).sum()), **paired(d[ii], groups[both][ii]))
                for panel,ii in panels.items()}
    return dict(rows={panel:len(both[ii]) for panel,ii in panels.items()}, correct=scores, transitions=transitions,
                final_step_positive_in_both=all(transitions[s+'-75-100']['combined']['net_correct']>0 for s in ORDERS),
                physical_fits=sum(len(v['fits']) for v in exports.values()), reused_supervised_heads=1,
                model_api_calls=0, new_rmb=0, course_validation_or_test=False)


def run(a):
    check(sha(a.training) == INPUT_TRAIN, 'Unexpected training packet')
    task = load_development_task(a.training)
    check(task.X_val is None, 'Training-only packet required')
    rows, y = task.X_train, np.asarray(task.y_train)
    groups, parts, _ = split(rows, a.first, a.second)
    old = read(a.control / 'completion.json')
    exp = read(a.control / 'exports.json')['regular']
    member = a.control / 'regular'
    check(old['protocol']==sha(a.control / 'protocol.json') and old['exports']==sha(a.control / 'exports.json')
          and exp['file']==sha(member / 'predictions.npy') and exp['fit']==sha(member / 'fit_completed.json')
          and read(member / 'fit_completed.json')['model']==sha(member / 'classifier.joblib'), 'Unbound E5 control')
    mainparts = read(a.control / 'protocol.json')['partitions']
    check(all(fingerprint(ii)==mainparts[k]['ids'] for k,ii in parts.items()), 'Registered partitions differ')
    sets = subsets(parts['train'], groups)
    a.out.mkdir(parents=True, exist_ok=True)
    frozen_save(a.out / 'protocol.json', dict(plan=sha(PLAN), runner=sha(Path(__file__)), control=sha(a.control/'completion.json'),
        input=INPUT_TRAIN, partitions=mainparts, fit_cap=9, model_api_cap=0, subsets={
            k:dict(rows=len(v), groups=len(set(groups[v])), row_order=fingerprint(v)) for k,v in sets.items()}))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    both = np.r_[parts['A'], parts['B']]
    exports = predict_all(a.out, sets, rows, y, both, a.control)
    result = summarize(a.out, sets, parts, groups, y, both, exports)
    check(result['physical_fits']==9, 'Unexpected fit count')
    frozen_save(a.out / 'results.json', result)
    frozen_save(a.out / 'completion.json', dict(results=sha(a.out/'results.json'), protocol=sha(a.out/'protocol.json'),
                                                exports=sha(a.out/'exports.json'), fits=9, model_api_calls=0))
    print(json.dumps(dict(correct=result['correct'], final_step_positive_in_both=result['final_step_positive_in_both'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','training','first','second','control','out'):
        p.add_argument('--'+name, type=lambda v:Path(v).resolve(), required=True)
    run(p.parse_args())
