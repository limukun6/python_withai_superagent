"""Test structural input and task-discovered local contexts for a frozen table model."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import GroupKFold

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.attribute_alignment import aligned_columns, comparable_fields, fitted_routes
from superagent.joint_context import blocks, tokens
from superagent.splits import sample_groups
from superagent.table_foundation import FrozenTableClassifier, VIEWS

PLAN=ROOT/"research/task1_feedback_extension_1127/contextual_inference.json"
OPTIONS=("global_structure","conditional_context")


def extend(x,profile,context,aligned):
    return np.column_stack((x,tokens(x[:,profile]),tokens(x[:,context]),aligned))


def train(out,name):
    info=read(out/"inputs.json")
    folder=out/("icl_"+name)
    folder.mkdir(exist_ok=False)
    training=bound(out,"fitting.joblib",info)
    x,y=training["x"],training["y"]
    query={f:bound(out,f+"_features.joblib",info) for f in ROLES}
    profile,context=blocks(x)
    families=comparable_fields(x,profile)
    require(families,"No recurring family with comparable values")
    routes=fitted_routes(x,y,families,context)
    save(folder/"launch.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),discovered_families=len(families)))
    meta={}
    counter=[]
    def fit(rows,labels):
        counter.append(dict(rows=len(labels),views=VIEWS))
        save(folder/("fit_"+str(len(counter))+".json"),counter[-1])
        return FrozenTableClassifier("categorical_tabular").fit(rows,labels)
    if name=="global_structure":
        group=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
        encoded=np.empty((len(y),2*len(families)),dtype=object)
        for own,held in GroupKFold(3).split(x,y,group):
            route=fitted_routes(x[own],y[own],families,context)
            encoded[held]=aligned_columns(x[held],route)
        model=fit(extend(x,profile,context,encoded),y)
        scores={f:model.predict_proba(extend(rows,profile,context,aligned_columns(rows,routes))) for f,rows in query.items()}
    else:
        gates=[g for g,route,_ in routes if g is not None and len(set(route.values()))>=2]
        require(gates,"No supported conditional gate")
        gate=max(gates,key=gates.count)
        raw=read(out/"trial_pretrained_icl/sealed.json")
        scores={}
        for f in ROLES:
            path=out/"trial_pretrained_icl"/(f+".npy")
            require(sha(path)==raw["predictions"][f],"Unbound global fallback")
            scores[f]=np.load(path,allow_pickle=False).copy()
        levels,support=np.unique(x[:,gate],return_counts=True)
        permitted=sorted([(int(n),v) for v,n in zip(levels,support,strict=True) if n>=128],reverse=True)[:6]
        rng=np.random.default_rng(61831)
        profiles=np.array(tokens(x[:,profile]))
        routed={f:0 for f in ROLES}
        for _,value in permitted:
            own=np.flatnonzero(x[:,gate]==value)
            other=np.flatnonzero(x[:,gate]!=value)
            if len(np.unique(y[own]))!=len(np.unique(y)):
                continue
            support_keys,counts=np.unique(profiles[own],return_counts=True)
            lookup=dict(zip(support_keys,counts))
            weight=np.array([1+np.sqrt(lookup.get(k,0)) for k in profiles[other]],dtype=float)
            contextual=rng.choice(other,min(len(other),len(own)//2),replace=False,p=weight/weight.sum())
            selected=np.sort(np.concatenate((own,contextual)))
            model=fit(x[selected],y[selected])
            for f,rows in query.items():
                mask=rows[:,gate]==value
                if mask.any():
                    scores[f][mask]=model.predict_proba(rows[mask])
                    routed[f]+=int(mask.sum())
        meta=dict(gate=int(gate),routed=routed,global_fallback_reused=True)
    seals={}
    for role,probs in scores.items():
        require(probs.shape==(len(query[role]),len(np.unique(y))) and np.isfinite(probs).all()
                and np.allclose(probs.sum(1),1,atol=1e-5),"Invalid probabilities")
        path=folder/(role+".npy")
        np.save(path,probs,allow_pickle=False)
        seals[role]=sha(path)
    result=dict(protocol=sha(PLAN),predictions=seals,physical_fits=len(counter),views=VIEWS*len(counter),
                context_sizes=[d["rows"] for d in counter],details=meta,api_requests=0)
    save(folder/"sealed.json",result)
    print(name,result,flush=True)


def elect(out):
    info=read(out/"inputs.json")
    folder=out/"icl_context_election"
    folder.mkdir(exist_ok=False)
    paths=dict(aligned=out/"trial_attribute_alignment",raw=out/"trial_pretrained_icl")
    paths.update({n:out/("icl_"+n) for n in OPTIONS})
    seals={k:read(p/"sealed.json") for k,p in paths.items()}
    classes=np.array(seals["aligned"]["classes"])
    def assess(role,options):
        bank={}
        for k,p in paths.items():
            f=p/(role+".npy")
            require(sha(f)==seals[k]["predictions"][role],"Changed probability bank")
            bank[k]=np.load(f,allow_pickle=False)
        y=bound(out,role+"_answers.joblib",info)
        ref=(bank["aligned"]+bank["raw"])/2
        hit=classes[ref.argmax(1)]==y
        records={}
        for n,weight in options:
            matrix=ref if n=="retain" else ref+weight*.5*(bank[n]-bank["raw"])
            records[n,weight]=paired(classes[matrix.argmax(1)]==y,hit)
        return records,len(y),int(hit.sum())
    options=[("retain",0.)]+[(n,w) for n in OPTIONS for w in (.5,1.)]
    panel,size,ref=assess("nomination",options)
    winner=min(options,key=lambda v:(-panel[v]["net"],v[1],v[0]))
    save(folder/"locked.json",dict(protocol=sha(PLAN),alternatives=[dict(candidate=n,weight=w,**v) for (n,w),v in panel.items()],
                            winner=list(winner),rows=size,reference=ref))
    later=[]
    for role in ROLES[1:]:
        candidate,n,base=assess(role,[winner])
        later.append(dict(role=role,rows=n,reference=base,**candidate[winner]))
    gain=sum(r["net"] for r in later)/sum(r["rows"] for r in later)
    result=dict(lock=sha(folder/"locked.json"),selected=list(winner),checks=later,gain=gain,
                eligible=min(r["net"] for r in later)>=0 and gain>=.003,api_requests=0)
    save(folder/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=(*OPTIONS,"elect"))
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    if args.action=="elect":
        elect(args.out)
    else:
        train(args.out,args.action)
