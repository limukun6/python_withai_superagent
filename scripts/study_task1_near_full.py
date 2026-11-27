"""Lock six ordinary solutions before grading the two remaining training-only folds."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_fusion_match import prepare
from scripts.study_task1_outer import compare,prediction_csv
from scripts.unified_released_study import ROOT,read,require,save,sha

PLAN=ROOT/"research/task1_feedback_extension_1127/near_full_confirmation.json"
ARMS=("prior","regular","near_full")
FOLDS=(3,4)


def grade(out):
    prepared=read(out/"prepared.json")
    signatures={}
    pred={}
    rows=[]
    for i in FOLDS:
        folder=out/str(i)
        info=read(folder/"inputs.json")
        require(prepared["folds"][str(i)]==sha(folder/"inputs.json") and sha(folder/"outer_answers.joblib")==info["outer_answers"],
                "Changed evaluation geometry")
        methods={}
        size=info["roles"]["external"]["rows"]
        for arm in ARMS:
            p=folder/arm
            r=read(p/"sealed.json")
            require(r["independent_reload"] and sha(p/"solve.csv")==r["prediction"]==sha(p/"reload.csv")
                    and r["status"]=="completed" and r["rows"]==size,"Incomplete ordinary entry")
            signatures[str((i,arm))]=sha(p/"sealed.json")
            pred[i,arm]=prediction_csv(p/"solve.csv",size)
            methods[arm]=dict(members=r["members"],fits=r["fits"])
        rows.append(dict(fold=i,rows=size,methods=methods,decision=read(folder/"near_full/run/agent/near_full_result.json")))
    save(out/"scoring_lock.json",dict(protocol=sha(PLAN),geometry=sha(out/"prepared.json"),ordinary=signatures))
    grouped=[]
    hits={a:[] for a in ARMS}
    for record in rows:
        i=record["fold"]
        key=joblib.load(out/str(i)/"outer_answers.joblib")
        truth=key["y"].astype(str)
        values={a:pred[i,a]==truth for a in ARMS}
        record["correct"]={a:int(h.sum()) for a,h in values.items()}
        record["paired"]={a:compare(values["near_full"],values[a],key["groups"]) for a in ARMS[:2]}
        for a in ARMS:
            hits[a].extend(values[a])
        grouped.extend(key["groups"])
    pooled={a:compare(np.asarray(hits["near_full"]),np.asarray(hits[a]),grouped) for a in ARMS[:2]}
    ok=all(min(row["paired"][a]["net"] for row in rows)>=0 and pooled[a]["gain"]>=.003 for a in ARMS[:2])
    result=dict(protocol=sha(PLAN),lock=sha(out/"scoring_lock.json"),rows=len(grouped),folds=rows,
         scores={a:dict(correct=int(np.sum(v)),accuracy=float(np.mean(v))) for a,v in hits.items()},
         comparisons=pooled,advance=bool(ok),api_requests=0,designated_validation_opened=False,official_test_opened=False)
    save(out/"result.json",result)
    print(result["scores"],pooled,ok,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","grade"))
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    parser.add_argument("--previous",type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    if a.action=="prepare":
        prepare(a.previous,a.out,indices=FOLDS,protocol=PLAN)
    else:
        grade(a.out)
