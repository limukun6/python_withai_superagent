"""Train a single trust adjustment from the earlier held folds of one ordinary run."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.study_disagreement_routing import correction
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, predictions, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask,load_prediction_features
from superagent.experiment_policy import identity
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.portfolio_review import evidence
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/pooled_disagreement_protocol.json"


def signals(base,extra,share):
    protected=base.argmax(axis=1)
    other=extra.argmax(axis=1)
    vote=((1-share)*base+share*extra).argmax(axis=1)
    a=np.abs(base[:,1]-base[:,0])
    b=np.abs(extra[:,1]-extra[:,0])
    offset=np.log(np.maximum(share*b,1e-12))-np.log(np.maximum((1-share)*a,1e-12))
    x=np.column_stack([np.ones(len(a)),np.log1p(10*a)/np.log(11),np.log1p(10*b)/np.log(11)])
    return dict(vote=vote,other=other,protected=protected,disagrees=other!=protected,x=x,offset=offset)


def adjusted(s,params):
    vote=s["vote"].copy()
    margin=s["offset"]+s["x"]@params
    choose_other=s["disagrees"]&(margin>0)
    choose_protected=s["disagrees"]&(margin<0)
    vote[choose_other]=s["other"][choose_other]
    vote[choose_protected]=s["protected"][choose_protected]
    return vote


def run(root,case):
    parent=root/("portfolio-recruitment" if case=="first" else "recruitment-replication")
    folder=parent/"comparison"
    arm=folder/"challenger"
    agent=arm/"run/agent"
    out=root/"pooled-disagreement"/case
    out.mkdir(parents=True,exist_ok=True)
    sealed=read(arm/"sealed.json")
    review=read(agent/"portfolio_review.json")
    require(sha(agent/"portfolio_review.json")==sealed["protected"]["review"],"Changed controller decision")
    lock(out/"inputs.json",dict(protocol=sha(PLAN),run=sha(arm/"sealed.json"),election=sealed["protected"]["review"]))
    if (review["status"]!="accepted" or not 0<review["nomination"]["share"]<1
            or sealed["supervised_fits"]>=read(parent/"inputs.json")["fit_cap"]):
        lock(out/"results.json",dict(eligible=False,reason=review["status"],extra_fits=0,model_api_requests=0))
        return
    data=joblib.load(agent/"refit_data.joblib")
    metric=MetricSpec.from_payload(data)
    require(list(metric.labels)==[0,1] and metric.name=="accuracy","Expected binary accuracy")
    incumbent=review["incumbent_members"]
    weights=review["incumbent_weights"]
    nominee=review["nomination"]["member"]
    share=review["nomination"]["share"]
    members=[*incumbent,nominee]
    history={(r["split_id"],r["candidate"]):agent/f"trial_{r['trial']}.joblib"
             for r in read(agent/"history.json") if r["status"]=="completed"}
    groups=sample_groups(TrainingTask(data["kind"],data["X_train"],data["y_train"]),"shared_text")
    mapping={text:g for row,g in zip(data["X_train"],groups) for text in row}
    used_groups=set()
    def panel(name):
        path=agent/(name+".joblib")
        payload=joblib.load(path)
        digest=sha(path)
        y=np.asarray(payload["y_search"])
        local=np.array([mapping[row[0]] for row in payload["X_search"]])
        require(all(mapping[a]==mapping[b] for a,b in payload["X_search"]) and not used_groups.intersection(local),
                "Scoring groups overlap across roles")
        bank={}
        for m in members:
            if name.startswith("review_"):
                file=agent/("portfolio_"+name+"_"+identity(m)[:20]+".joblib")
            else:
                file=history.get((name,m),agent/f"analogy_{name}_{m}.joblib")
            if not file.exists():
                return None
            obj=joblib.load(file)
            facts=obj["metrics"]
            require(facts["resolved_configuration"]["id"]==m,"Different fitted method")
            bank[m]=planner_observation(obj,facts,dict(sha256=digest,labels=y),metric)
        used_groups.update(local)
        base=evidence(bank,incumbent,weights,metric)
        extra=evidence(bank,[nominee],[1.],metric)
        require(base["row_key"]==extra["row_key"],"Different prediction rows")
        return signals(base["probabilities"],extra["probabilities"],share),y,local
    observed=[]
    for name in ("primary","confirmation_1","confirmation_2"):
        trial=panel(name)
        if trial is not None:
            observed.append(trial)
    if observed:
        x=np.concatenate([v["x"][v["disagrees"]] for v,_,_ in observed])
        off=np.concatenate([v["offset"][v["disagrees"]] for v,_,_ in observed])
        target=np.concatenate([(v["other"]==y)[v["disagrees"]] for v,y,_ in observed]).astype(float)
        fit_groups=np.unique(np.concatenate([g[v["disagrees"]] for v,_,g in observed]))
    if len(observed)<2 or len(fit_groups)<80 or len(set(target))!=2:
        lock(out/"results.json",dict(eligible=False,reason="insufficient matched disagreements",training_folds=len(observed),
                                     extra_fits=0,model_api_requests=0))
        return
    fit_path=out/"fitted.json"
    if not fit_path.exists():
        with (out/"fit_started.json").open("x") as f:
            f.write('{"attempted":1}\n')
        params=correction(x,off,target)
        lock(fit_path,dict(inputs=sha(out/"inputs.json"),folds=len(observed),train_rows=sum(len(y) for _,y,_ in observed),
                           disagreements=len(target),independent_groups=len(fit_groups),params=params.tolist()))
    params=np.asarray(read(fit_path)["params"])
    evidence_folds=[]
    for name in ("review_1","review_2"):
        s,y,_=panel(name)
        evidence_folds.append(dict(name=name,rows=len(y),**paired(adjusted(s,params)==y,s["vote"]==y)))
    allow=(min(r["net"] for r in evidence_folds)>=0 and sum(r["net"] for r in evidence_folds)/
           sum(r["rows"] for r in evidence_folds)>=.002)
    lock(out/"decision.json",dict(fit=sha(fit_path),checks=evidence_folds,adopt=allow))
    activate_resource_directory(root/"arctic-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"feature_bank"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    record=read(agent/"final_selection.json")
    file=agent/record["model_file"]
    require(sha(file)==sealed["model"] and record["ensemble_components"]==members,"Outer model differs from the sealed election")
    models=joblib.load(file)["model"].models
    query=load_prediction_features(folder/"q.zip",data["kind"],read(agent/"task_spec.json"))
    bank=[]
    for model in models:
        order,prob=probability_with_model(data["kind"],model,query)
        bank.append(prob[:,[list(order).index(c) for c in metric.labels]])
    s=signals(np.average(bank[:-1],axis=0,weights=weights),bank[-1],share)
    require(sha(arm/"reload.csv")==sealed["predictions"] and
            np.array_equal(s["vote"],predictions(arm/"reload.csv",len(query))),"Unmodified vote differs from the cold run")
    shadow=adjusted(s,params)
    decided=shadow if allow else s["vote"]
    dig=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    lock(out/"outer_lock.json",dict(decision=sha(out/"decision.json"),fixed=dig(s["vote"]),shadow=dig(shadow),deployed=dig(decided)))
    truth=folder/"truth.joblib"
    require(sha(truth)==read(parent/"inputs.json")["cases"]["comparison"]["truth"],"Different audit labels")
    labels=joblib.load(truth)["labels"]
    lock(out/"results.json",dict(eligible=True,outer_lock=sha(out/"outer_lock.json"),checks=evidence_folds,
         adopt=allow,rows=len(labels),fixed_correct=int((s["vote"]==labels).sum()),
         deployed=paired(decided==labels,s["vote"]==labels),shadow=paired(shadow==labels,s["vote"]==labels),
         training_folds=len(observed),disagreements=len(target),disagreement_groups=len(fit_groups),
         extra_fits=1,base_fits=0,model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(read(out/"results.json"),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True,choices=("first","second"))
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    run(args.root,args.case)
