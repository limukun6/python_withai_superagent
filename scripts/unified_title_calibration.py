"""Test a single calibrated weight for a frozen two-member preference classifier on training data."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit

from scripts.unified_released_study import PLAN, ROOT, env, read, require, save, sha
from superagent.ab_complement import canonical_probabilities
from superagent.data import load_prediction_features
from superagent.metrics import MetricSpec
from superagent.score_evidence import checked_score_record


def logits(probability):
    return logit(np.clip(probability, 1e-7, 1-1e-7))


def fit(agent, members, out):
    records=read(agent/'prediction_evidence.json')
    splits=[v['split_id'] for v in records if v['split_id']!='primary']
    split=next(s for s in reversed(splits) if all(any(r['candidate_id']==n and r['split_id']==s for r in records) for n in members))
    payload=agent/(split+'.joblib')
    data=joblib.load(payload)
    y=np.asarray(data['y_search'])
    metric=MetricSpec.from_payload(data)
    columns=[]
    for name in members:
        row=next(r for r in records if r['candidate_id']==name and r['split_id']==split)
        require(row['evaluation_payload_sha256']==sha(payload),'Wrong confirmation data')
        values,classes=checked_score_record(row['score_record'],row,metric,name,y)
        columns.append(logits(values[:,classes.index(1)]))
    def objective(weight):
        z=weight*columns[0]+(1-weight)*columns[1]
        return float(np.mean(np.logaddexp(0.,z)-y*z)+.02*(weight-.5)**2)
    result=minimize_scalar(objective,method='bounded',bounds=(0,1),options={'xatol':1e-10})
    require(result.success,'Weight estimation failed')
    save(out/'weight.json',dict(members=members,weight=float(result.x),fit_rows=len(y),split=split,
         source_payload=sha(payload),objective_at_half=objective(.5),fit_objective=float(result.fun),
         supervised_fits=1,base_model=sha(agent/'selected_model.joblib'),plan=sha(PLAN),runner=sha(Path(__file__))))


def infer(agent, study, out):
    weight=read(out/'weight.json')
    require(sha(agent/'selected_model.joblib')==weight['base_model'],'Base model changed')
    features=study/'development_features.zip'
    require(sha(features)==read(study/'protocol.json')['development_features'],'Wrong development features')
    pairs=load_prediction_features(features,'pair_text',read(agent/'task_spec.json'))
    model=joblib.load(agent/'selected_model.joblib')['model']
    require(len(model.models)==2,'Two models required for the registered calibration')
    values=np.stack([canonical_probabilities(m,pairs)[1][:,1] for m in model.models],axis=1)
    tie=np.array([a<b for a,b in pairs])
    def choose(z):
        return ((z>0)|((z==0)&tie)).astype(int)
    baseline=choose(values.mean(axis=1)-.5)
    w=weight['weight']
    challenge=choose(logits(values)@np.array([w,1-w]))
    require(np.array_equal(baseline,model.predict(pairs)),'Equal-weight control differs from the actual model')
    target=out/'development_predictions.csv'
    with target.open('x',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['row_id','control','calibrated'])
        writer.writerows((i,int(x),int(y)) for i,(x,y) in enumerate(zip(baseline,challenge)))
    save(out/'predictions_frozen.json',dict(rows=len(pairs),features=sha(features),weights=sha(out/'weight.json'),
                                            predictions=sha(target)))


def score(study,out):
    seal=read(out/'predictions_frozen.json')
    target=out/'development_predictions.csv'
    truth=study/'development_truth.json'
    require(sha(target)==seal['predictions'] and sha(truth)==read(study/'protocol.json')['development_truth'],'Sealed inputs differ')
    with target.open() as f:
        rows=list(csv.DictReader(f))
    require([r['row_id'] for r in rows]==list(map(str,range(seal['rows']))),'Row order differs')
    data=read(truth)
    y=np.array(data['labels'])
    base=np.array([int(r['control']) for r in rows])==y
    new=np.array([int(r['calibrated']) for r in rows])==y
    _,groups=np.unique(data['groups'],return_inverse=True)
    sizes=np.bincount(groups)
    delta=np.bincount(groups,weights=new.astype(int)-base.astype(int)).astype(int)
    rng=np.random.default_rng(28113)
    bootstrap=[]
    for _ in range(2000):
        sample=rng.integers(0,len(sizes),size=len(sizes))
        bootstrap.append(delta[sample].sum()/sizes[sample].sum())
    interval=np.quantile(bootstrap,[.025,.975]).tolist()
    gained,lost=int((new&~base).sum()),int((base&~new).sum())
    save(out/'grouped_deltas.json',dict(rows=sizes.tolist(),correct_difference=delta.tolist()))
    save(out/'result.json',dict(scope='historical training-only development reserve',rows=len(y),groups=len(sizes),
         reference_correct=int(base.sum()),calibrated_correct=int(new.sum()),gained=gained,lost=lost,
         paired_interval_fraction=interval,promoted=bool(gained-lost>=5 and interval[0]>0),predictions=seal['predictions'],
         supervised_fit_count=1,new_api_calls=0,designated_validation_used=False,official_test_used=False))
    print(json.dumps(read(out/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=lambda v:Path(v).resolve())
    p.add_argument('--weights',required=True,type=lambda v:Path(v).resolve())
    p.add_argument('--historical-agent',type=lambda v:Path(v).resolve())
    p.add_argument('--infer',action='store_true')
    a=p.parse_args()
    out=a.study/'calibration'
    require((a.study/'scores.json').exists(),'Complete the unified baseline first')
    agent=a.study/'titles/agent'
    members=read(agent/'final_selection.json')['ensemble_components']
    if len(members)!=2:
        require(a.historical_agent is not None,'The registered fallback requires the immutable historical pair')
        agent=a.historical_agent
        members=read(agent/'final_selection.json')['ensemble_components']
    if a.infer:
        infer(agent,a.study,out)
    else:
        out.mkdir(exist_ok=True)
        if not (out/'weight.json').exists():
            fit(agent,members,out)
        if not (out/'predictions_frozen.json').exists():
            cmd=[sys.executable,'-m','scripts.unified_title_calibration',str(a.study),'--weights',str(a.weights),'--infer']
            if a.historical_agent:
                cmd+=['--historical-agent',str(a.historical_agent)]
            subprocess.run(cmd,cwd=ROOT,env=env(a.weights,out/'unexpected_fit.jsonl'),check=True)
        score(a.study,out)
