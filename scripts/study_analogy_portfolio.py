"""Measure complementarity to the frozen autonomous ensemble using existing supervised predictions."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.analogy_review import nominate,promoted
from superagent.autonomous_run import planner_observation
from superagent.finalist_race import observation
from superagent.metrics import MetricSpec


PLAN=ROOT/'research/paired_encoder_search_1127/analogy_portfolio_protocol.json'


def run(root):
    output=root/'analogy-portfolio-complement'
    output.mkdir(exist_ok=True)
    base=root/'analogy-raw/original/run/agent'
    old=root/'analogy-raw/revised/run/agent'
    more=root/'analogy-close-confirm'
    chosen=read(base/'finalist_result.json')
    members=chosen['selected']
    seal=read(root/'analogy-raw/original/sealed.json')
    require(seal['independent_reload'] and seal['members']==members and len(members)>1,'Missing elected portfolio')
    adviser=read(old/'analogy_review.json')['candidate']
    history=read(base/'history.json')
    reference=next(s['scores'] for s in chosen['comparisons'] if s['members']==members)
    panels=[]
    for fold in ('primary','confirmation_1','confirmation_2'):
        paths={}
        for member in members:
            trial=next(r['trial'] for r in history if r['candidate']==member and r['split_id']==fold and r['status']=='completed')
            paths[member]=base/f'trial_{trial}.joblib'
        paths[adviser]=old/f'analogy_primary_{adviser}.joblib' if fold=='primary' else more/f'{fold}_{adviser}.joblib'
        payload=base/(fold+'.joblib')
        expected=read(base/(fold+'_split.json'))['payload_sha256']
        require(expected==sha(payload)==read(old/(fold+'_split.json'))['payload_sha256'],'Changed payload or rows')
        panels.append((fold,payload,paths,expected))
    lock(output/'inputs.json',dict(source=sha(PLAN),portfolio=sha(base/'finalist_result.json'),
         adviser_pilot=sha(more/'results.json'),panels=[dict(fold=f,payload_sha256=h,components={k:sha(v) for k,v in paths.items()}) for f,_,paths,h in panels]))
    results=[]
    decision=None
    for fold,payload,paths,digest in panels:
        data=joblib.load(payload)
        metric=MetricSpec.from_payload(data)
        truth=np.asarray(data['y_search'])
        records={}
        for name,path in paths.items():
            obj=joblib.load(path)
            require(obj['metrics']['resolved_configuration']['id']==name,'Unexpected fitted method')
            records[name]=planner_observation(obj,obj['metrics'],dict(sha256=digest,labels=truth),metric)
        anchor=observation(records,members,metric)
        require(abs(anchor['score']-reference[fold])<1e-12,'Could not reproduce the portfolio score')
        near=records[adviser]
        if fold=='primary':
            decision=nominate(anchor,near,truth)
            lock(output/'nomination.json',dict(inputs=sha(output/'inputs.json'),decision=decision))
        weight=decision['weight']
        base_right=np.asarray(anchor['predictions'])==truth
        blend=(1-weight)*np.asarray(anchor['probabilities'])+weight*np.asarray(near['probabilities'])
        vote_right=np.asarray(metric.labels)[blend.argmax(axis=1)]==truth
        plus=int((~base_right&vote_right).sum())
        minus=int((base_right&~vote_right).sum())
        results.append(dict(fold=fold,rows=len(truth),anchor=int(base_right.sum()),chosen=int(vote_right.sum()),gained=plus,lost=minus,gain=plus-minus))
    passed=bool(decision['weight'] and promoted(results[0]['gain'],results[1:],primary_rows=results[0]['rows']))
    lock(output/'results.json',dict(lock=sha(output/'nomination.json'),decision=decision,folds=results,advance=passed,
         new_classifier_fits=0,new_model_requests=0,course_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(decision,results,passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args().root)
