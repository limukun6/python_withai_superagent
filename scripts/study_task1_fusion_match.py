"""Create and assess three disjoint training-only comparisons of the autonomous blend policy."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_extensions import key, npy, seeded
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.unified_released_study import ROOT, make_zip, read, require, save, sha
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/fusion_holdout.json"
ARMS=("regular","separate","prior")
REUSE=("weighted","replicas")


def prepare(previous,out, *, indices=(0,1,2), protocol=PLAN, seed_label="AIAA new training partition blend qualification v1"):
    original=previous/"anonymous/training.zip"
    require(sha(original)==read(previous/"inputs.json")["entry_inputs"],"Origin changed")
    x,y=arrays(original,["C_train.npy","y_train.npy"])
    group=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    seed=seeded(seed_label)
    out.mkdir(exist_ok=False)
    used=set()
    for i,(fit,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(x,y,group)):
        if i not in indices:
            continue
        folder=out/str(i)
        packets=folder/"anonymous"
        packets.mkdir(parents=True)
        groups=set(group[held])
        require(not used.intersection(groups) and not groups.intersection(group[fit]),"Training groups overlap")
        used.update(groups)
        train=packets/"training.zip"
        query=packets/"queries.zip"
        make_zip(train,[("C_train.npy",npy(x[fit])),("y_train.npy",npy(y[fit])),
                       ("Readme.md",b"Binary categorical classification. The primary metric is accuracy.")])
        make_zip(query,[("C_test.npy",npy(x[held]))])
        answer=folder/"outer_answers.joblib"
        joblib.dump(dict(y=y[held],groups=group[held]),answer)
        save(folder/"inputs.json",dict(protocol=sha(protocol),source=sha(original),seed=seed,
             entry_inputs=sha(train),entry_queries=sha(query),outer_answers=sha(answer),
             roles=dict(external=dict(rows=len(held),groups=len(groups),digest=key(held)),
                        fit=dict(rows=len(fit),digest=key(fit)))))
    save(out/"prepared.json",dict(protocol=sha(protocol),source=sha(original),folds={str(i):sha(out/str(i)/"inputs.json")
                for i in indices},unique_evaluation_groups=len(used)))


def grade(out):
    inputs=read(out/"prepared.json")
    locked={}
    predictions={}
    summary=[]
    for i in range(3):
        fold=out/str(i)
        info=read(fold/"inputs.json")
        require(sha(fold/"inputs.json")==inputs["folds"][str(i)] and sha(fold/"outer_answers.joblib")==info["outer_answers"],
                "Changed training partition")
        methods={}
        for arm in ARMS:
            path=fold/arm
            seal=read(path/"sealed.json")
            require(seal["independent_reload"] and sha(path/"reload.csv")==sha(path/"solve.csv")==seal["prediction"]
                    and seal["status"]=="completed" and seal["rows"]==info["roles"]["external"]["rows"],"Unsealed solution")
            predictions[i,arm]=prediction_csv(path/"solve.csv",seal["rows"])
            locked[str((i,arm))]=sha(path/"sealed.json")
            methods[arm]=dict(members=seal["members"],fits=seal["fits"])
        reuse=fold/"reuse"
        extra=read(reuse/"sealed.json")
        require(extra["old_prediction_rebuilt"] and extra["rows"]==info["roles"]["external"]["rows"],
                "Unverified reconstruction")
        locked[str((i,"reuse"))]=sha(reuse/"sealed.json")
        for arm in REUSE:
            p=reuse/(arm+".npy")
            require(sha(p)==extra["predictions"][arm],"Changed cached-inference predictions")
            predictions[i,arm]=np.load(p,allow_pickle=False).astype(str)
        guard=read(fold/"separate/run/agent/fusion_check.json")
        summary.append(dict(fold=i,methods=methods,guard=guard,reuse=read(reuse/"decision.json")))
    save(out/"score_lock.json",dict(inputs=sha(out/"prepared.json"),results=locked,protocol=sha(PLAN),
        additional_inference_protocol=sha(PLAN.with_name("inference_reuse.json")),
        unchanged_control_protocol=sha(PLAN.with_name("three_fold_anchor.json"))))
    total={a:[] for a in (*ARMS,*REUSE)}
    groups=[]
    for i,row in enumerate(summary):
        answer=joblib.load(out/str(i)/"outer_answers.joblib")
        y=answer["y"].astype(str)
        scores={arm:predictions[i,arm]==y for arm in total}
        row.update(rows=len(y),correct={a:int(v.sum()) for a,v in scores.items()},
                   paired=compare(scores["separate"],scores["regular"],answer["groups"]),
                   vs_prior={a:compare(scores[a],scores["prior"],answer["groups"]) for a in ("regular","separate")},
                   diagnostic_comparisons={a:compare(scores[a],scores["regular"],answer["groups"]) for a in REUSE})
        for a in total:
            total[a].extend(scores[a])
        groups.extend(answer["groups"])
    d=compare(np.array(total["separate"]),np.array(total["regular"]),groups)
    against_prior={a:compare(np.array(total[a]),np.array(total["prior"]),groups) for a in ("regular","separate")}
    def qualifies(gains,pooled):
        return sum(delta>0 for delta in gains)>=2 and min(gains)>=-.003 and pooled["gain"]>=.003
    regular_pass=qualifies([row["vs_prior"]["regular"]["gain"] for row in summary],against_prior["regular"])
    guard_vs_regular=qualifies([row["paired"]["gain"] for row in summary],d)
    guard_vs_prior=qualifies([row["vs_prior"]["separate"]["gain"] for row in summary],against_prior["separate"])
    release="separate" if guard_vs_regular and guard_vs_prior else "regular" if regular_pass else None
    result=dict(score_lock=sha(out/"score_lock.json"),folds=summary,pooled=d,
        correct={a:int(np.sum(x)) for a,x in total.items()},rows=len(groups),
        diagnostic_comparisons={a:compare(np.array(total[a]),np.array(total["regular"]),groups) for a in REUSE},
        against_prior=against_prior,guard_vs_regular_qualified=guard_vs_regular,guard_vs_prior_qualified=guard_vs_prior,
        regular_qualified=regular_pass,advance=release,
        api_requests=0,designated_validation_opened=False,official_test_opened=False)
    save(out/"result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","grade"])
    parser.add_argument("--previous",type=lambda s:Path(s).resolve())
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    if a.action=="prepare":
        prepare(a.previous,a.out)
    else:
        grade(a.out)
