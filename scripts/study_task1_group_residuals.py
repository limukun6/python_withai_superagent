"""Estimate persistent log-odds residuals from repeated groups and out-of-group tree predictions."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, logit

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/group_residuals.json"
METHODS=("main","interaction")


def keys(x,columns):
    return np.asarray(tokens(x[:,columns]))


def fit_effect(key,y,p,sides):
    vocabulary,ix=np.unique(key,return_inverse=True)
    count=np.bincount(ix,minlength=len(vocabulary))
    info=[]
    signal=[]
    for side in (0,1):
        select=sides==side
        signal.append(np.bincount(ix[select],weights=(y-p)[select],minlength=len(vocabulary)))
        info.append(np.bincount(ix[select],weights=(p*(1-p))[select],minlength=len(vocabulary)))
    n=float(np.dot(info[0],info[1]))
    variance=float(np.clip(np.dot(signal[0],signal[1])/n,0,1.5)) if n>0 else 0.
    shared=int(((info[0]>0)&(info[1]>0)).sum())
    if shared<8 or not variance:
        return {},0.,dict(shared_groups=shared,prior_variance=0.)
    delta=(signal[0]+signal[1])/(info[0]+info[1]+1/variance)
    delta-=np.average(delta,weights=count)
    strength=float(np.average(delta**2,weights=count))
    return dict(zip(vocabulary,delta)),strength,dict(shared_groups=shared,prior_variance=variance,
                                                    rms_offset=float(np.sqrt(strength)))


def apply(table,key):
    return np.fromiter((table.get(k,0.) for k in key),dtype=float,count=len(key))


def run(out):
    info=read(out/"inputs.json")
    fit=bound(out,"fitting.joblib",info)
    x,raw=fit["x"],fit["y"]
    classes,y=np.unique(raw,return_inverse=True)
    teacher=out/"distill_teachers"
    description=read(teacher/"sealed.json")
    require(sha(teacher/"out_of_fold.npy")==description["predictions"] and description["inputs"]==sha(out/"inputs.json")
            and description["overlap"]==0 and description["rows"]==len(y),"Unverified teacher feedback")
    p=np.load(teacher/"out_of_fold.npy",allow_pickle=False)[:,0]
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    halves=np.array([hashlib.sha256(str(g).encode()).digest()[0]%2 for g in groups])
    profile,context=blocks(x)
    groupings=[tuple(profile),tuple(context)]
    for name in METHODS:
        current=logit(np.clip(p,1e-6,1-1e-6))
        effects=[]
        summary=[]
        for columns in groupings:
            grouped=keys(x,columns)
            table,_,record=fit_effect(grouped,y,expit(current),halves)
            effects.append((columns,table))
            current+=apply(table,grouped)
            summary.append(record)
        if name=="interaction":
            ranked=[]
            for j in context:
                if 2<=len(np.unique(x[:,j]))<=24:
                    columns=(*profile,j)
                    table,strength,record=fit_effect(keys(x,columns),y,expit(current),halves)
                    ranked.append((strength,columns,table,record))
            if ranked:
                _,columns,table,record=max(ranked,key=lambda value:value[0])
                effects.append((columns,table))
                summary.append(record)
        folder=out/("groupoffset_"+name)
        folder.mkdir(exist_ok=False)
        source=out/"trial_anchor"
        anchor=read(source/"sealed.json")
        assert anchor["classes"]==classes.tolist()
        signatures={}
        for role in ROLES:
            rows=bound(out,role+"_features.joblib",info)
            file=source/(role+".npy")
            require(sha(file)==anchor["predictions"][role],"Unbound base probabilities")
            z=logit(np.clip(np.load(file,allow_pickle=False)[:,1],1e-6,1-1e-6))
            for columns,table in effects:
                z+=apply(table,keys(rows,columns))
            probability=expit(z)
            file=folder/(role+".npy")
            np.save(file,np.column_stack((1-probability,probability)),allow_pickle=False)
            signatures[role]=sha(file)
        save(folder/"sealed.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),classes=classes.tolist(),
             predictions=signatures,grouping=len(effects),fit_diagnostics=summary,correction_fits=1,
             borrowed_teacher=sha(teacher/"sealed.json"),api_requests=0))
        print(name,summary,flush=True)


def grade(out):
    elect(out,METHODS,"groupoffset",PLAN)
    info=read(out/"inputs.json")
    record={}
    for role in ROLES:
        labels=bound(out,role+"_answers.joblib",info)
        bank={}
        for key in ("trial_anchor",*("groupoffset_"+s for s in METHODS)):
            source=out/key
            d=read(source/"sealed.json")
            p=source/(role+".npy")
            require(sha(p)==d["predictions"][role],"Changed standalone comparison")
            bank[key]=np.asarray(d["classes"])[np.load(p,allow_pickle=False).argmax(1)]==labels
        record[role]={s:paired(bank["groupoffset_"+s],bank["trial_anchor"]) for s in METHODS}
    save(out/"groupoffset_election/vs_original.json",record)
    print(record,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    (run if a.action=="run" else grade)(a.out)
