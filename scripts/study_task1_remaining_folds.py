"""Qualify the 48-fit confirmation policy on new cold anonymous outer folds from the full training set."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.run_task1_profile_screen import load_split
from scripts.study_task1_extensions import npy
from scripts.study_task1_fusion_match import prepare as make_folds
from scripts.study_task1_outer import compare, prediction_csv
from scripts.unified_released_study import ROOT, make_zip, read, require, save, sha

PLAN=ROOT/"research/task1_feedback_extension_1127/remaining_fold_qualification.json"
FOLDS=(0,1,2)
ARMS=("partner","regular","extended")


def prepare(archive,out):
    origin=out.with_name(out.name+"_source")
    (origin/"anonymous").mkdir(parents=True,exist_ok=False)
    x,y,access=load_split(archive,"train")
    require(len(y)==len(x),"Invalid training package")
    path=origin/"anonymous/training.zip"
    make_zip(path,[("C_train.npy",npy(x.astype(str))),("y_train.npy",npy(y)),
             ("Readme.md",b"Binary categorical classification. The declared primary metric is accuracy.")])
    save(origin/"inputs.json",dict(training_members=access,entry_inputs=sha(path),rows=len(y),raw_archive=sha(archive)))
    make_folds(origin,out,indices=FOLDS,protocol=PLAN,seed_label="AIAA full training extended evidence panels v1")


def signature(folder):
    entries=read(folder/"run/agent/history.json")
    return [(r["candidate"],r["split_id"],r["status"],r.get("search_score")) for r in entries]


def grade(out):
    inputs=read(out/"prepared.json")
    require(inputs["protocol"]==sha(PLAN),"Changed qualification protocol")
    prior,regular,extended=ARMS
    decisions=[]
    sealed={}
    predictions={}
    for i in FOLDS:
        root=out/str(i)
        info=read(root/"inputs.json")
        require(inputs["folds"][str(i)]==sha(root/"inputs.json") and sha(root/"outer_answers.joblib")==info["outer_answers"],
                "Unexpected outer split")
        rows=info["roles"]["external"]["rows"]
        arm_info={}
        for arm in ARMS:
            side=root/arm
            record=read(side/"sealed.json")
            launch=read(side/"launch.json")
            require(launch["cap"]==48 and record["launch"]==sha(side/"launch.json") and record["status"]=="completed"
                    and record["rows"]==rows and record["model_api_requests"]==0 and record["independent_reload"]
                    and sha(side/"solve.csv")==record["prediction"]==sha(side/"reload.csv"),"Incomplete cold solution")
            predictions[i,arm]=prediction_csv(side/"solve.csv",rows)
            sealed[str((i,arm))]=sha(side/"sealed.json")
            arm_info[arm]=dict(fits=record["fits"],members=record["members"],source=launch["source"])
        require(arm_info[regular]["source"]==arm_info[extended]["source"],"Unmatched expanded solvers")
        review=read(root/extended/"run/agent/remaining_fold_result.json")
        decisions.append(dict(fold=i,rows=rows,methods=arm_info,review=review,
                 identical_initial_search=signature(root/regular)==signature(root/extended)))
    save(out/"scoring_lock.json",dict(protocol=sha(PLAN),geometry=sha(out/"prepared.json"),solves=sealed))
    hits={a:[] for a in ARMS}
    groups=[]
    for item in decisions:
        i=item["fold"]
        answer=joblib.load(out/str(i)/"outer_answers.joblib")
        y=answer["y"].astype(str)
        correct={a:predictions[i,a]==y for a in ARMS}
        item["scores"]={a:int(c.sum()) for a,c in correct.items()}
        item["paired"]={a:compare(correct[extended],correct[a],answer["groups"]) for a in (prior,regular)}
        for a,c in correct.items():
            hits[a].extend(c)
        groups.extend(answer["groups"])
    pooled={a:compare(np.asarray(hits[extended]),np.asarray(hits[a]),groups) for a in (prior,regular)}
    qualifies=all(pooled[a]["gain"]>=.003 and min(r["paired"][a]["net"] for r in decisions)>=0 for a in (prior,regular))
    result=dict(protocol=sha(PLAN),lock=sha(out/"scoring_lock.json"),folds=decisions,rows=len(groups),
                correct={a:int(np.sum(hits[a])) for a in ARMS},pooled=pooled,advance=bool(qualifies),
                model_api_requests=0,designated_validation_opened=False,official_test_opened=False)
    save(out/"result.json",result)
    print(result["correct"],pooled,qualifies,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","grade"))
    for p in ("out","archive"):
        parser.add_argument("--"+p,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    if a.action=="prepare":
        prepare(a.archive,a.out)
    else:
        grade(a.out)
