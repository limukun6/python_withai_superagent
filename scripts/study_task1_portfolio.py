"""Fit a conservative probability portfolio from existing feedback and evaluate group-held decisions."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_evidence_allocation import PREFIXES, load_bank
from scripts.study_task1_outer import compare
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/portfolio_stacking.json"
PENALTIES=(.0005,.005,.05)


def logloss(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def train(y,base,changes,penalty):
    def objective(weights):
        p=np.clip(base+changes@weights,1e-8,1-1e-8)
        loss=logloss(y,p)+penalty*(weights@weights+weights.sum()**2)
        grad=changes.T@((p-y)/(p*(1-p)))/len(y)+2*penalty*(weights+weights.sum())
        return loss,grad
    n=changes.shape[1]
    constraint=dict(type="ineq",fun=lambda w:.5-w.sum(),jac=lambda w:-np.ones(n))
    fit=minimize(objective,np.zeros(n),method="SLSQP",jac=True,bounds=[(0,.5)]*n,
                 constraints=[constraint],options=dict(maxiter=240,ftol=1e-10))
    require(fit.success and np.min(fit.x)>=-1e-8 and fit.x.sum()<=.5+1e-7,"Invalid portfolio optimizer result")
    return fit.x


def run(out):
    data=out/"development"
    bank,seals,base,classes=load_bank(data,(*PREFIXES,"typedicl_"))
    names=sorted(bank)
    changes=np.column_stack([bank[n]-base for n in names])
    info=read(data/"inputs.json")
    x=np.concatenate([bound(data,r+"_features.joblib",info) for r in ROLES])
    y=np.concatenate([bound(data,r+"_answers.joblib",info) for r in ROLES])==classes[1]
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    outer=StratifiedGroupKFold(6,shuffle=True,random_state=583257)
    locked=[]
    predictions={"protected":np.full(len(y),np.nan),"unguarded":np.full(len(y),np.nan)}
    coverage=np.zeros(len(y),dtype=int)
    decisions=[]
    fits=0
    for k,(fit,held) in enumerate(outer.split(x,y,groups)):
        require(not set(groups[held])&set(groups[fit]),"Shared group in outer router fitting")
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=91211+k)
        scored={p:[] for p in PENALTIES}
        deltas={p:[] for p in PENALTIES}
        sizes=[]
        for a,b in inner.split(x[fit],y[fit],groups[fit]):
            a,b=fit[a],fit[b]
            require(not set(groups[a])&set(groups[b]),"Shared group in inner router fitting")
            sizes.append(len(b))
            for penalty in PENALTIES:
                weights=train(y[a],base[a],changes[a],penalty)
                fits+=1
                pred=base[b]+changes[b]@weights
                scored[penalty].append(logloss(y[b],pred)-logloss(y[b],base[b]))
                deltas[penalty].append(int(((pred>.5)==y[b]).sum()-((base[b]>.5)==y[b]).sum()))
        best=min(PENALTIES,key=lambda v:np.average(scored[v],weights=sizes))
        cutoff=np.average(scored[best],weights=sizes)+np.std(scored[best],ddof=1)/np.sqrt(len(sizes))
        chosen=max(v for v in PENALTIES if np.average(scored[v],weights=sizes)<=cutoff)
        gain=sum(deltas[chosen])/sum(sizes)
        admit=(gain>=.003 and sum(n>0 for n in deltas[chosen])>=2 and np.average(scored[chosen],weights=sizes)<0)
        w=train(y[fit],base[fit],changes[fit],chosen)
        fits+=1
        p=base[held]+changes[held]@w
        predictions["unguarded"][held]=p
        predictions["protected"][held]=p if admit else base[held]
        coverage[held]+=1
        locked.append(dict(fold=k,admit=bool(admit),penalty=chosen,inner_net=deltas[chosen],inner_log_loss_differences=scored[chosen],
              retained_weight=float(1-w.sum()),other_weights={m:float(v) for m,v in zip(names,w) if v>1e-6}))
        decisions.append(held)
    require((coverage==1).all() and all(np.isfinite(p).all() for p in predictions.values()),"Unsealed portfolio rows")
    dest=out/"portfolio_stacking"
    dest.mkdir(exist_ok=False)
    checksums={}
    for n,p in predictions.items():
        path=dest/(n+".npy")
        np.save(path,p,allow_pickle=False)
        checksums[n]=sha(path)
    save(dest/"locked.json",dict(protocol=sha(PLAN),training_only_seals=seals,decisions=locked,predictions=checksums))
    ref=(base>.5)==y
    hits={n:(p>.5)==y for n,p in predictions.items()}
    comparisons={n:compare(v,ref,groups) for n,v in hits.items()}
    episodes=[dict(fold=i,rows=len(ix),ref=int(ref[ix].sum()),
                   **{n:int(v[ix].sum()) for n,v in hits.items()}) for i,ix in enumerate(decisions)]
    passes=comparisons["protected"]["gain"]>=.003 and sum(r["protected"]>=r["ref"] for r in episodes)>=4
    record=dict(protocol=sha(PLAN),lock=sha(dest/"locked.json"),models=len(names),rows=len(y),router_fits=fits,base_fits=0,
                admitted=sum(r["admit"] for r in locked),episodes=episodes,ref=int(ref.sum()),correct={n:int(v.sum()) for n,v in hits.items()},
                comparisons=comparisons,advance=bool(passes),api_requests=0)
    save(dest/"result.json",record)
    print(record,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    with threadpool_limits(limits=2):
        run(parser.parse_args().out)
