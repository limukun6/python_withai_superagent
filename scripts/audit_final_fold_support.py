"""Check how much training support a different final family had on the six graded development tasks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.conditional_review import group_observation
from superagent.metrics import MetricSpec


def values(metric,y,predicted,indices):
    labels=np.unique(y)
    outcome,pred=y[indices],predicted[indices]
    if metric=='accuracy':
        return (outcome==pred).mean(axis=1)
    scores=[]
    for label in labels:
        positive=outcome==label
        tp=(positive & (pred==label)).sum(axis=1)
        if metric=='balanced_accuracy':
            scores.append(tp/positive.sum(axis=1))
        elif metric=='macro_f1':
            scores.append(2*tp/(positive.sum(axis=1)+(pred==label).sum(axis=1)))
        else:
            raise ValueError('Unregistered diagnostic metric')
    return np.mean(scores,axis=0)


def check(folder):
    agent=folder/'control/run/agent'
    final=read(agent/'final_selection.json')
    incumbent=read(agent/'verified_selection.json')['single_candidate']
    members=final['ensemble_components'] or [final['single_candidate']]
    if members==[incumbent]:
        return dict(changed_family=False)
    metric=MetricSpec.from_payload(read(agent/'task_spec.json'))
    history=read(agent/'history.json')
    rng=np.random.default_rng(261177)
    deltas,sizes,draws={},[],[]
    folds=('primary','confirmation_1','confirmation_2')
    held_rows=set()
    for fold in folds:
        path=agent/(fold+'.joblib')
        split=read(agent/(fold+'_split.json'))
        require(sha(path)==split['payload_sha256'] and not held_rows.intersection(split['held']),'Mismatched or overlapping holdouts')
        held_rows.update(split['held'])
        y=joblib.load(path)['y_search']
        payload=dict(labels=y,sha256=sha(path))
        bank={}
        for name in dict.fromkeys([*members,incumbent]):
            r=next(r for r in history if r['status']=='completed' and r['candidate']==name and r['split_id']==fold)
            bundle=joblib.load(agent/f"trial_{r['trial']}.joblib")
            bank[name]=planner_observation(bundle,bundle['metrics'],payload,metric)
        current=bank[incumbent]
        contender=group_observation(bank,members,metric)
        require(abs(contender['score']-final['fold_scores'][fold])<1e-9,'Training scores disagree with the actual selection')
        deltas[fold]=contender['score']-current['score']
        fixed=np.arange(len(y))[None,:]
        def delta(indices):
            return values(metric.name,y,contender['predictions'],indices)-values(metric.name,y,current['predictions'],indices)
        require(abs(float(delta(fixed)[0])-deltas[fold])<1e-9,'Bootstrap metric does not reproduce point estimate')
        by_class=[np.flatnonzero(y==lab) for lab in np.unique(y)]
        samples=np.concatenate([rng.choice(ids,(5000,len(ids)),replace=True) for ids in by_class],axis=1)
        sizes.append(len(y))
        draws.append(delta(samples))
    d=np.array(list(deltas.values()))
    leave_one=[float(np.average(np.delete(d,i),weights=np.delete(sizes,i))) for i in range(len(folds))]
    return dict(changed_family=True,metric=metric.name,fold_gains=deltas,rows=sizes,
                weighted_gain=float(np.average(d,weights=sizes)),leave_one_fold_out=leave_one,
                descriptive_paired_95=np.quantile(np.average(draws,axis=0,weights=sizes),[.025,.975]).tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('main','scheduled','output'):
        p.add_argument(k,type=lambda x:Path(x).resolve())
    p.add_argument('--weights',type=lambda x:Path(x).resolve(),required=True)
    a=p.parse_args()
    if os.environ.get('AB_B1_OFFLINE')=='1':
        paths=[a.main/c for c in ('s51','s52','s53','s54')]+[a.scheduled/c for c in ('n74','n75')]
        result={folder.name:check(folder) for folder in paths}
        save(a.output,dict(tasks=result,method='Retrospective training-only comparison of the actual final family versus the last verified primary checkpoint. Per-fold class-stratified paired bootstrap, 5000 draws with seed 261177. The interval describes row variation with models fixed and does not account for adaptive selection or training-set overlap.',
                           new_fits=0,new_api_calls=0,outer_labels_read=False))
    else:
        fitlog=a.scheduled/'unexpected_fold_audit_fits.jsonl'
        subprocess.run([sys.executable,'-m','scripts.audit_final_fold_support',*sys.argv[1:]],cwd=ROOT,env=env(a.weights,fitlog),check=True)
        require(not fitlog.exists() or not fitlog.read_text().strip(),'New fit in a training-evidence audit')
