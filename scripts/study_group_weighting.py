"""Test whether repeated comparisons should carry less supervised training weight."""
import argparse
from copy import copy
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/group_weighting_protocol.json"
FOLDS=("primary","confirmation_1","confirmation_2")


def run(root):
    out=root/"group-weighting"
    out.mkdir(exist_ok=True)
    raw=root/"analogy-portfolio-raw/portfolio_revised"
    agent=raw/"run/agent"
    review=read(agent/"analogy_review.json")
    final=read(agent/"final_selection.json")
    require(review["status"]=="accepted" and final["model_sha256"]==read(raw/"sealed.json")["model_sha256"],
            "Unbound source election")
    source=review["source_member"]
    parts=dict(zip(final["ensemble_components"],final["weights"],strict=True))
    history=read(agent/"history.json")
    paths={}
    for fold in FOLDS:
        found={r["candidate"]:r["trial"] for r in history if r["status"]=="completed" and r["split_id"]==fold}
        paths[fold]={m:(agent/f"trial_{found[m]}.joblib" if m in found else agent/f"analogy_{fold}_{m}.joblib") for m in parts}
    lock(out/"inputs.json",dict(protocol=sha(PLAN),election=sha(raw/"sealed.json"),review=sha(agent/"analogy_review.json"),
         member=source,share=parts[source],panels={fold:dict(payload=sha(agent/(fold+".joblib")),
             constituents={m:sha(p) for m,p in paths[fold].items()}) for fold in FOLDS}))
    activate_resource_directory(root/"discovery-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"feature_bank"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    physical=out/"physical.jsonl"
    def panel(fold, estimates):
        folder=out/fold
        folder.mkdir(exist_ok=True)
        data=joblib.load(agent/(fold+".joblib"))
        digest=sha(agent/(fold+".joblib"))
        require(digest==read(agent/(fold+"_split.json"))["payload_sha256"],"Changed internal panel")
        metric=MetricSpec.from_payload(data)
        classes=np.asarray(metric.labels)
        y=np.asarray(data["y_train"])
        truth=np.asarray(data["y_search"])
        original={}
        for member,path in paths[fold].items():
            stored=joblib.load(path)
            observation=planner_observation(stored,stored["metrics"],dict(sha256=digest,labels=truth),metric)
            original[member]=np.asarray(observation["probabilities"])
        baseline=sum(w*original[m] for m,w in parts.items())
        parent=joblib.load(paths[fold][source])["model"]
        estimator=parent.named_steps["classifier"]
        require(isinstance(estimator,LogisticRegression) and list(classes)==[0,1],"Unsupported parent")
        train=data["X_train"]
        groups=sample_groups(TrainingTask("pair_text",train,y),"shared_text")
        _,row_group,size=np.unique(groups,return_inverse=True,return_counts=True)
        weight=1/np.sqrt(size[row_group])
        weight/=np.mean(weight)
        x=parent.named_steps["features"].transform(train)
        x=vstack([x,-x],format="csr")
        mirrored=np.r_[y,1-y]
        def refit(mode,weights):
            buffer=folder/(mode+".npy")
            receipt=folder/(mode+".json")
            if receipt.exists():
                require(sha(buffer)==read(receipt)["prediction_sha256"],"Stored fit does not match output")
                return np.load(buffer,allow_pickle=False)
            tag=dict(fold=fold,mode=mode)
            prior=[json.loads(s) for s in physical.read_text().splitlines()] if physical.exists() else []
            require(tag not in prior and len(prior)<4,"Existing in-flight fit or budget exceeded")
            new=clone(estimator)
            with physical.open("a") as stream:
                stream.write(json.dumps(tag)+"\n")
            new.fit(x,mirrored,sample_weight=weights)
            model=copy(parent)
            model.steps=[*parent.steps[:-1],(parent.steps[-1][0],new)]
            probability=model.predict_proba(data["X_search"])
            probability=probability[:,[list(model.classes_).index(c) for c in classes]]
            np.save(buffer,probability,allow_pickle=False)
            lock(receipt,dict(payload=digest,source=sha(paths[fold][source]),mode=mode,prediction_sha256=sha(buffer)))
            return probability
        error=None
        if fold=="primary":
            reproduced=refit("uniform",None)
            error=float(np.max(np.abs(reproduced-original[source])))
            require(error<=.000002,"Uniform refit does not reproduce its recorded parent")
        candidate=refit("weighted",np.tile(weight,2))
        base_correct=classes[baseline.argmax(axis=1)]==truth
        measurements={}
        for share in estimates:
            blend=baseline+share*parts[source]*(candidate-original[source])
            measurements[share]=paired(classes[blend.argmax(axis=1)]==truth,base_correct)
        return dict(fold=fold,train_rows=len(y),held_rows=len(truth),training_groups=len(size),max_group_rows=int(size.max()),
                    weight_range=[float(weight.min()),float(weight.max())],uniform_max_error=error,
                    base_correct=int(base_correct.sum()),shares={str(k):v for k,v in measurements.items()})
    first=panel("primary",(0.,.5,1.))
    share=min((0.,.5,1.),key=lambda w:(-first["shares"][str(w)]["correct"],w))
    lock(out/"nomination.json",dict(inputs=sha(out/"inputs.json"),primary=first,selected_share=share))
    confirmations=[]
    if share>0:
        for fold in FOLDS[1:]:
            record=panel(fold,(share,))
            confirmations.append(record)
            lock(out/(fold+".json"),record)
            if record["shares"][str(share)]["net"]<0:
                break
    deltas=[r["shares"][str(share)]["net"] for r in confirmations]
    net=first["shares"][str(share)]["net"]+sum(deltas)
    count=first["held_rows"]+sum(r["held_rows"] for r in confirmations)
    advance=len(deltas)==2 and min(deltas)>=0 and max(deltas)>0 and net/count>=.002
    fits=len(physical.read_text().splitlines())
    require(fits==2+len(confirmations),"Unexpected supervised fit count")
    lock(out/"results.json",dict(nomination=sha(out/"nomination.json"),primary=first,selected_share=share,
         confirmations=confirmations,net=net,evaluated_rows=count,advance=advance,fits=fits,model_api_requests=0,
         inferred_labels=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(first,share,deltas,advance,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    with threadpool_limits(limits=2):
        run(parser.parse_args().root)
