"""Measure one fixed bag of frozen training-subset models on its registered comparison rows."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.package_b_grouped_titles import masks
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_training_subset_vote_1123_zh.md'
WEIGHTS = {'full':.5, 'curve-a-75':.25, 'curve-b-75':.25}


def original_head(folder):
    model = folder/'model.joblib'
    fitted = read(folder/'fit_completed.json')
    check(fitted['model']==sha(model) and fitted['start']==sha(folder/'fit_started.json'), 'Head identity differs')
    return joblib.load(model)


def labels(probs, rows):
    return ((probs>.5) | ((probs==.5) & np.array([a<b for a,b in rows]))).astype(int)


def run(a):
    prior = read(a.curve/'completion.json')
    cp = read(a.curve/'protocol.json')
    check(prior['exports']==sha(a.curve/'exports.json') and prior['protocol']==sha(a.curve/'protocol.json')
          and cp['control']==sha(a.control/'completion.json') and sha(a.training)==cp['input']==INPUT_TRAIN,
          'Unbound input or train-size result')
    task = load_development_task(a.training)
    rows = task.X_train
    check(task.X_val is None,'Expected training-only packet')
    groups, parts, _ = split(rows,a.first,a.second)
    check(all(cp['partitions'][k]['ids']==fingerprint(v) for k,v in parts.items()), 'Partition mismatch')
    both = np.r_[parts['A'],parts['B']]
    search = [rows[i] for i in both]
    a.out.mkdir(parents=True,exist_ok=True)
    frozen_save(a.out/'protocol.json',dict(runner=sha(Path(__file__)),plan=sha(PLAN),input=INPUT_TRAIN,
        curve=sha(a.curve/'completion.json'),weights=WEIGHTS,fit_cap=0,model_api_cap=0,partitions=cp['partitions']))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    exports = read(a.curve/'exports.json')
    raw = {}
    for key in WEIGHTS:
        folder = a.curve/key
        exp = exports[key]
        for name,sealed in exp['fits'].items():
            check(sha(folder/name/'fit_completed.json')==sealed,'Original head changed')
        if key=='full':
            reg = a.control/'regular'
            meta = read(a.control/'exports.json')['regular']
            check(read(a.control/'completion.json')['exports']==sha(a.control/'exports.json') and
                  meta['fit']==sha(reg/'fit_completed.json') and
                  read(reg/'fit_completed.json')['model']==sha(reg/'classifier.joblib'),'Original full E5 mismatch')
            features = AlternativeSentenceFeatures('pair_text','e5').fit([rows[i] for i in parts['train']])
            e5 = Pipeline([('features',features),('classifier',joblib.load(reg/'classifier.joblib'))])
        else:
            e5 = original_head(folder/'e5')
        fam = original_head(folder/'familiarity')
        ensemble = PreferenceProbabilityVote('pair_text',[e5,fam],np.array([0,1]))
        raw[key] = ensemble.predict_proba(search)[:,1]
        prior_preds = folder/'predictions.npy'
        check(sha(prior_preds)==exp['file'],'Frozen curve predictions changed')
        np.testing.assert_array_equal(labels(raw[key],search),np.load(prior_preds,allow_pickle=False))
        sample = search[:23]
        forward = ensemble.predict_proba(sample)[:,1]
        rev = ensemble.predict_proba([(r,l) for l,r in sample])[:,1]
        np.testing.assert_allclose(rev,1-forward,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(labels(forward,sample),labels(raw[key][:23],sample))
    combined = sum(WEIGHTS[k]*p for k,p in raw.items())
    base, new = labels(raw['full'],search), labels(combined,search)
    file = a.out/'predictions.npz'
    if not file.exists():
        with file.open('xb') as f:
            np.savez_compressed(f, control=base, mixture=new, control_probability=raw['full'], mixture_probability=combined)
    with np.load(file) as stored:
        for pred,key in [(base,'control'),(new,'mixture'),(raw['full'],'control_probability'),(combined,'mixture_probability')]:
            np.testing.assert_array_equal(pred,stored[key])
    frozen_save(a.out/'exports.json',dict(file=sha(file),row_order=fingerprint(both),control=fingerprint(base),
                                          mixture=fingerprint(new),reused_classifier_heads=6))
    truth = np.asarray(task.y_train)[both]
    hits = base==truth
    win = new==truth
    delta = win.astype(int)-hits.astype(int)
    results = {}
    for panel,ix in [('A',slice(0,len(parts['A']))),('B',slice(len(parts['A']),None)),('combined',slice(None))]:
        d=delta[ix]
        results[panel]=dict(rows=len(d),baseline=int(hits[ix].sum()),mixture=int(win[ix].sum()),
                            gained=int((d>0).sum()),lost=int((d<0).sum()),delta=paired(d,groups[both][ix]))
    gains=[results[k]['mixture']-results[k]['baseline'] for k in ('A','B')]
    public=masks(['control','mixture'],dict(control=base,mixture=new),truth,groups[both],
                 ['A']*len(parts['A'])+['B']*len(parts['B']))
    frozen_save(a.out/'grouped_correctness.json',public)
    record=dict(panels=results,eligible=min(gains)>=0 and sum(gains)>=8,physical_fits=0,model_api_calls=0,new_rmb=0,
                course_validation_or_test=False)
    frozen_save(a.out/'results.json',record)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
                exports=sha(a.out/'exports.json'),groups=sha(a.out/'grouped_correctness.json'),fits=0,model_api_calls=0))
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','training','first','second','control','curve','out'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
