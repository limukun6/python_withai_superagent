"""Fixed LightGBM/number-interpretation ablation, trained on course training rows only."""
import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.joint_context import blocks, tokens
from superagent.splits import sample_groups

CONFIGS = dict(native=dict(num_leaves=31,n_estimators=1000,learning_rate=.035,min_child_samples=30,reg_lambda=5),
               ordered_values=dict(num_leaves=31,n_estimators=1000,learning_rate=.035,min_child_samples=30,reg_lambda=5),
               regularized=dict(num_leaves=15,n_estimators=1400,learning_rate=.025,min_child_samples=50,reg_lambda=10))


def number(value):
    v=str(value).lower().strip().replace('$','').replace(',','')
    if v in ('__missing__','nan','none','null','?'):
        return np.nan
    if v=='never':
        return 0.
    clock=re.fullmatch(r'(\d{1,2})(?::(\d\d))?\s*(am|pm)',v)
    if clock:
        return int(clock[1])%12 + (12 if clock[3]=='pm' else 0) + int(clock[2] or 0)/60
    duration=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(h|d|min)',v)
    if duration:
        return float(duration[1])*{'h':1,'d':24,'min':1/60}[duration[2]]
    interval=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:-|~)\s*(\d+(?:\.\d+)?)',v)
    if interval:
        return (float(interval[1])+float(interval[2]))/2
    one=re.fullmatch(r'(less|below|under|gt|above|over)?\s*(\d+(?:\.\d+)?)\s*(plus|or more)?',v)
    if one:
        offset=-.5 if one[1] in ('less','below','under') else .5 if one[1] or one[3] else 0
        return float(one[2])+offset
    raise ValueError('No supported numeric interpretation')


def matrix(a,b):
    profile,context=blocks(a)
    def augment(x):
        return np.column_stack((x,tokens(x[:,profile]),tokens(x[:,context])))
    enc=OrdinalEncoder(dtype=np.float32,handle_unknown='use_encoded_value',unknown_value=-1)
    x=enc.fit_transform(augment(a))+1
    q=enc.transform(augment(b))+1
    numeric=[]
    for j in range(a.shape[1]):
        try:
            [number(v) for v in np.unique(a[:,j])]
        except ValueError:
            continue
        numeric.append(j)
    extra_train=np.array([[number(v) for v in row[numeric]] for row in a],dtype=np.float32)
    extra_query=[]
    for row in b:
        values=[]
        for j in numeric:
            try:
                values.append(number(row[j]))
            except ValueError:
                values.append(np.nan)
        extra_query.append(values)
    return x,q,extra_train,np.array(extra_query,dtype=np.float32),numeric


def run(task,out):
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    x,y,hashes=load_split(task,'train')
    groups=sample_groups(SimpleNamespace(kind='categorical_tabular',X_train=x))
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=18643).split(x,y,groups))[:2]
    root=Path(__file__).resolve().parents[1]
    ref_path=root/'research/task1_joint_context_01'
    refs=json.loads((ref_path/'results.json').read_text(encoding='utf-8'))
    ref_protocol=json.loads((ref_path/'protocol.json').read_text(encoding='utf-8'))
    save(out/'protocol.json',dict(members=hashes,configs=CONFIGS,lightgbm=lgb.__version__,
        runner_sha=digest(Path(__file__).read_bytes()),fit_cap=6,api_calls=0,test_opened=False,validation_opened=False,
        model_sources={p.name:digest(p.read_bytes()) for p in (root/'superagent/joint_context.py',
                           root/'superagent/profile_history.py',root/'superagent/splits.py')},
        baseline_results_sha=digest((ref_path/'results.json').read_bytes())))
    rows=[]
    fits=0
    try:
        for fold,(train,held) in enumerate(splits):
            if digest(held.astype('<i8').tobytes())!=ref_protocol['splits'][fold]['held_sha']:
                raise ValueError('Reference fold mismatch')
            a,b,ae,be,numeric=matrix(x[train],x[held])
            record=dict(fold=fold,held=len(held),numeric_columns=numeric,held_sha=digest(held.astype('<i8').tobytes()))
            for name,config in CONFIGS.items():
                tick=time.monotonic()
                fits+=1
                save(out/'last_attempt.json',dict(fold=fold,method=name,admitted_fits=fits))
                model=lgb.LGBMClassifier(**config,objective='binary',n_jobs=6,verbosity=-1,
                    random_state=26471,deterministic=True,force_col_wise=True,cat_l2=10,cat_smooth=20)
                aa,bb=(a,b) if name=='native' else (np.column_stack((a,ae)),np.column_stack((b,be)))
                model.fit(aa,y[train],categorical_feature=list(range(a.shape[1])))
                pred=model.predict(bb)
                correct=int(np.sum(pred==y[held]))
                accuracy=correct/len(held)
                record[name]=dict(correct=correct,accuracy=accuracy,
                    versus_tokens_pp=100*(accuracy-refs[fold]['tokens']['accuracy']),
                    versus_native_pp=100*(accuracy-record.get('native',dict(accuracy=accuracy))['accuracy']),
                    seconds=round(time.monotonic()-tick,3),prediction_sha=digest(np.asarray(pred,dtype='<i8').tobytes()))
                save(out/'results.json',rows+[record])
                print(fold,name,record[name],flush=True)
            rows.append(record)
        summary={}
        for name in CONFIGS:
            delta=[r[name]['versus_tokens_pp'] for r in rows]
            summary[name]=dict(mean_delta_pp=float(np.mean(delta)),promote=bool(min(delta)>0 and np.mean(delta)>=.5))
        save(out/'completion.json',dict(status='complete',fits=fits,summary=summary,api_calls=0,tokens=0,
                                        course_yuan=0,seconds=round(time.monotonic()-start,3)))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',admitted_fits=fits,error=repr(exc),api_calls=0,
                                        seconds=round(time.monotonic()-start,3)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.task,args.output)
