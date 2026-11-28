"""Test whether histories explain responses beyond population-level offer difficulty."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.run_task1_exposure import paired
from scripts.run_task1_profile_screen import digest, load_split, save
from scripts.task1_exposure_models import context_axis
from superagent.field_interactions import InteractionTree, choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, tokens
from superagent.models import _tabular_catboost
from superagent.splits import sample_groups


def population_tree(x, y):
    model = InteractionTree(dict(pair_limit=3,backbone_params=dict(depth=6,l2_leaf_reg=3.)))
    model.pairs = choose_pairs(rank_pairs(x,y),3)
    features = model.transform(x)
    model.classifier = _tabular_catboost(y,6,3.)
    model.classifier.set_params(thread_count=6)
    model.classifier.fit(features,y,cat_features=list(range(features.shape[1])))
    model.classes_ = model.classifier.classes_
    return model


def estimate_offsets(axis, keys, y, expected):
    levels, codes = np.unique(axis,return_inverse=True)
    design = np.column_stack((np.ones(len(y)),np.eye(len(levels))[codes]))
    precision = np.r_[1/.75**2,np.full(len(levels),1/1.25**2)]
    offsets = {}
    groups = {}
    for i,k in enumerate(keys):
        groups.setdefault(k,[]).append(i)
    with threadpool_limits(limits=1):
        for key,ids in groups.items():
            predictors = design[ids]
            beta = np.zeros(len(precision))
            base = logit(np.clip(expected[ids],1e-6,1-1e-6))
            for _ in range(16):
                p = expit(base + predictors @ beta)
                grad = predictors.T @ (p-y[ids]) + precision*beta
                hess = predictors.T @ ((p*(1-p))[:,None]*predictors) + np.diag(precision)
                step = np.linalg.solve(hess,grad)
                beta -= step
                if np.max(np.abs(step)) < 1e-8:
                    break
            offsets[key] = beta
    return {level:j+1 for j,level in enumerate(levels)},offsets


def run(task,out):
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    root=Path(__file__).resolve().parents[1]
    x,y,members=load_split(task,'train')
    groups=sample_groups(SimpleNamespace(X_train=x,kind='categorical_tabular'))
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=112801).split(x,y,groups))
    refs=json.loads((root/'research/task1_exposure_01/results.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),root/'scripts/task1_exposure_models.py',root/'scripts/run_task1_exposure.py',
             root/'superagent/field_interactions.py',root/'superagent/joint_context.py',
             root/'docs/batches/TASK1_PERSONAL_OFFSETS_PLAN_20261128_zh.md']
    save(out/'protocol.json',dict(sources={p.name:digest(p.read_bytes()) for p in sources},members=members,
         folds=[digest(b.astype('<i8').tobytes()) for _,b in folds],primary_tree_cap=15,seed=112801,api_calls=0))
    fits=0
    estimated=0
    rows=[]
    winner=None
    try:
        for fold,(train,held) in enumerate(folds):
            if fold>=3 and winner is None:
                break
            def enter(name):
                nonlocal fits
                fits+=1
                save(out/'last_attempt.json',dict(fold=fold,model=name,tree_admissions=fits))
            enter('context_reference')
            ref_model=ContextTokenTree(dict(pair_limit=3)).fit(x[train],y[train])
            ref=ref_model.predict_proba(x[held])[:,1]
            uncertain=np.abs(ref-.5)<=.15
            baseline=paired(ref,ref,y[held],uncertain)
            if fold<len(refs):
                assert baseline['prediction_sha']==refs[fold]['methods']['baseline']['prediction_sha']
            profile,context=ref_model.profile,ref_model.context
            axis=context_axis(x[train],y[train],context)
            enter('population_all')
            population=population_tree(x[train],y[train]).predict_proba(x[held])[:,1]
            keys=np.asarray(tokens(x[train][:,profile]))
            cross=StratifiedGroupKFold(3,shuffle=True,random_state=4987)
            expected=np.full(len(train),np.nan)
            for i,(fit,oof) in enumerate(cross.split(x[train],y[train],keys)):
                assert not set(keys[fit]) & set(keys[oof])
                enter(f'other_profiles_{i}')
                model=population_tree(x[train[fit]],y[train[fit]])
                expected[oof]=model.predict_proba(x[train[oof]])[:,1]
            assert np.isfinite(expected).all()
            levels,offsets=estimate_offsets(x[train,axis],keys,y[train],expected)
            estimated+=len(offsets)
            correction=np.zeros(len(held))
            for i,(key,value) in enumerate(zip(tokens(x[held][:,profile]),x[held,axis],strict=True)):
                if key in offsets:
                    b=offsets[key]
                    correction[i]=b[0]+(b[levels[value]] if value in levels else 0.)
            adjusted=expit(logit(np.clip(population,1e-6,1-1e-6))+correction)
            row=dict(fold=fold,held=len(held),profile_estimates=len(offsets),axis=int(axis),
                     methods={'baseline':baseline})
            for name,prob in [('population',population),('adjusted',adjusted),('blend',(adjusted+ref)/2)]:
                row['methods'][name]=paired(prob,ref,y[held],uncertain)
            rows.append(row)
            save(out/'results.json',rows)
            print(fold,{n:v['correct'] for n,v in row['methods'].items()},flush=True)
            if fold==2:
                eligible={}
                for name in ('adjusted','blend'):
                    gains=[r['methods'][name]['delta_pp'] for r in rows]
                    gain_adjusted=sum(r['methods']['adjusted']['correct']-r['methods']['population']['correct'] for r in rows)
                    if min(gains)>=-.5 and np.mean(gains)>=.5 and sum(g>0 for g in gains)>=2 and gain_adjusted>0:
                        eligible[name]=float(np.mean(gains))
                winner=max(eligible,key=eligible.get,default=None)
                save(out/'screen.json',dict(winner=winner,eligible=eligible))
        net=[r['methods'][winner]['correct']-r['methods']['baseline']['correct'] for r in rows] if winner else []
        save(out/'completion.json',dict(status='complete',tree_fits=fits,profile_estimates=estimated,
            winner=winner,confirmed=bool(winner and sum(net[3:])>0 and 100*sum(net)/sum(r['held'] for r in rows)>=.5),
            seconds=round(time.monotonic()-start,3),api_calls=0,course_yuan=0,validation_opened=False,test_opened=False))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',tree_admissions=fits,profile_estimates=estimated,error=repr(exc),api_calls=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.task,args.output)
