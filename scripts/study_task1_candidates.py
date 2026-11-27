"""Fit each predeclared table candidate with access only to fitting labels, then elect on held roles."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.study_task1_extensions import PLAN,ROLES
from scripts.unified_released_study import read,require,save,sha
from superagent.joint_context import ContextTokenTree

METHODS=("anchor","attribute_alignment","pretrained_icl")
DIGEST="bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def bound(root,key,info):
    path=root/key
    require(sha(path)==info["packets"][key],"Changed research packet")
    return joblib.load(path)


def train(out,method):
    src=read(out/"inputs.json")
    folder=out/("trial_"+method)
    folder.mkdir(exist_ok=True)
    fit=bound(out,"fitting.joblib",src)
    x,y=fit["x"],fit["y"]
    classes=np.unique(y)
    require(len(classes)==2,"Requires two classes")
    device="cpu"
    metadata={}
    if method=="pretrained_icl":
        from importlib.metadata import version
        import torch
        from tabicl import TabICLClassifier
        require(version("tabicl")=="2.0.3","Changed downstream model library")
        probe=read(out.parent/"tabicl-device.json")
        device=probe["selected"]
        checkpoint=out.parent/"tabicl-classifier-v2-20260212.ckpt"
        require(device in ("mps","cpu") and sha(checkpoint)==DIGEST,"Unverified device or weight file")
        torch.set_num_threads(3)
        model=TabICLClassifier(n_estimators=4,batch_size=1,model_path=checkpoint,allow_auto_download=False,
                    device=device,use_amp=False,use_fa3=False,kv_cache=True,n_jobs=1,random_state=173)
        metadata=dict(checkpoint_sha256=DIGEST,ensemble_members=4,fit_type="frozen pretrained in-context classifier",
                      device=device,training_context_rows=len(y))
    elif method=="anchor":
        model=ContextTokenTree(dict(pair_limit=3))
    elif method=="attribute_alignment":
        from superagent.attribute_alignment import AttributeAlignmentTree
        model=AttributeAlignmentTree(dict(pair_limit=3))
    else:
        raise ValueError("Unknown comparison method")
    save(folder/"inputs.json",dict(protocol=sha(PLAN),packets=sha(out/"inputs.json"),method=method,**metadata))
    start=time.monotonic()
    with (folder/"attempt.json").open("x") as stream:
        json.dump(dict(physical_fit_calls=1,member_units=4 if method=="pretrained_icl" else 1,
                       subset="fixed fitting rows"),stream)
    try:
        model.fit(x,y)
        train_seconds=round(time.monotonic()-start,3)
        if method=="attribute_alignment":
            metadata.update(model.audit_)
        if method!="pretrained_icl":
            joblib.dump(model,folder/"model.joblib",compress=3)
        require(set(model.classes_)==set(classes),"Changed model class domain")
        columns=[list(model.classes_).index(c) for c in classes]
        signatures={}
        role_timings={}
        for role in ROLES:
            # Future role features may be predicted with the frozen fit; their labels remain in separate files.
            t=time.monotonic()
            query=bound(out,role+"_features.joblib",src)
            probs=np.asarray(model.predict_proba(query),dtype=float)[:,columns]
            require(probs.shape==(len(query),len(classes)) and np.isfinite(probs).all()
                    and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-5),"Invalid probabilities")
            path=folder/(role+".npy")
            np.save(path,probs,allow_pickle=False)
            signatures[role]=sha(path)
            role_timings[role]=round(time.monotonic()-t,3)
        record=dict(phase="sealed",inputs=sha(folder/"inputs.json"),training=len(y),classes=classes.tolist(),
                    predictions=signatures,model_sha256=sha(folder/"model.joblib") if method!="pretrained_icl" else None,
                    physical_fits=1,fit_seconds=train_seconds,role_seconds=role_timings,details=metadata,
                    model_api_requests=0,designated_validation_opened=False,official_test_opened=False)
        save(folder/"sealed.json",record)
        print(method,"sealed",record["fit_seconds"],role_timings,metadata,flush=True)
    except Exception as exc:
        save(folder/"failure.json",dict(error=type(exc).__name__,reason=str(exc)[:1200],seconds=round(time.monotonic()-start,3),
                                        member_units_reserved=4 if method=="pretrained_icl" else 1))
        raise


def paired(proposed,base):
    return dict(correct=int(proposed.sum()),fixed=int((proposed & ~base).sum()),
                lost=int((base & ~proposed).sum()),net=int(proposed.sum())-int(base.sum()))


def elect(out):
    src=read(out/"inputs.json")
    refs={m:read(out/("trial_"+m)/"sealed.json") for m in METHODS}
    classes=np.array(refs["anchor"]["classes"])
    require(all(d["classes"]==classes.tolist() and d["phase"]=="sealed" for d in refs.values()),
            "All candidates need sealed predictions in the same class order")
    def panel(role,options):
        bank={}
        for m in set(m for group,_ in options for m in group)|{"anchor"}:
            path=out/("trial_"+m)/(role+".npy")
            require(sha(path)==refs[m]["predictions"][role],"Prediction changed")
            bank[m]=np.load(path,allow_pickle=False)
        truth=bound(out,role+"_answers.joblib",src)
        ref=bank["anchor"]
        base=classes[ref.argmax(1)]==truth
        result={}
        for group,weight in options:
            prob=ref if not group else (1-weight)*ref+weight*sum(bank[m] for m in group)/len(group)
            correct=classes[prob.argmax(1)]==truth
            result[group,weight]=dict(**paired(correct,base),probability_digest=hashlib.sha256(prob.tobytes()).hexdigest())
        return dict(role=role,rows=len(truth),baseline=int(base.sum())),result
    choices=[((),0.)]+[(g,p) for n in (1,2) for g in itertools.combinations(METHODS[1:],n) for p in (.125,.25,.5,1.)]
    first,measured=panel("nomination",choices)
    winner=min(choices,key=lambda o:(-measured[o]["correct"],o[1],len(o[0]),measured[o]["probability_digest"]))
    folder=out/"election"
    folder.mkdir(exist_ok=True)
    save(folder/"nomination.json",dict(protocol=sha(PLAN),seals={m:sha(out/("trial_"+m)/"sealed.json") for m in METHODS},
         **first,alternatives=[dict(members=list(g),share=p,**v) for (g,p),v in measured.items()],
         selected=list(winner[0]),share=winner[1]))
    later=[]
    if winner[1]:
        for role in ROLES[1:]:
            info,scores=panel(role,[winner]);record=dict(**info,**scores[winner]);later.append(record)
            save(folder/(role+".json"),record)
            if record["net"]<0: break
    gain=sum(v["net"] for v in later)/max(1,sum(v["rows"] for v in later))
    eligible=len(later)==2 and min(v["net"] for v in later)>=0 and gain>=.003
    result=dict(nomination=sha(folder/"nomination.json"),selected=list(winner[0]),share=winner[1],checks=later,
                confirmation_gain=gain,advance=eligible,model_api_requests=0,outer_scored=False,
                designated_validation_opened=False,official_test_opened=False)
    save(folder/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("train","elect"));p.add_argument("--method",choices=METHODS)
    p.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    a=p.parse_args()
    if a.action=="elect":elect(a.out)
    else:train(a.out,a.method)
