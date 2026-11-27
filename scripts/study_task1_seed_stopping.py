"""Evaluate a train-only decision to spend the remaining fit budget on two more seeds."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_replicas import probabilities, reseed, SEEDS
from scripts.study_task1_threshold import get_fold
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.metrics import MetricSpec
from superagent.score_evidence import checked_score_record
from superagent.seed_replication import evidence
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/seed_stopping.json"
NEXT=(301147,814961)
SPLIT="confirmation_2"


def three(replica, tag, original):
    values=[original]
    for seed in SEEDS:
        path=replica/f"{tag}_{seed}.npy"
        if path.exists():
            require(sha(path)==read(path.with_suffix(".json"))["probabilities"],"Changed prior seed")
            values.append(np.load(path,allow_pickle=False))
    require(len(values) in (1,3),"Incomplete three-seed control")
    if len(values)==1:
        values *= 3
    return values


def extend(root, tag, model, data, y, x, classes, previous):
    values=list(previous)
    for seed in NEXT:
        candidate=reseed(model,seed)
        if candidate is None:
            values.append(previous[0])
            continue
        key=f"{tag}_{seed}"
        save(root/(key+"_attempt.json"),dict(protocol=sha(PLAN),seed=seed,requested_classifier_fits=1))
        candidate.fit(data,y)
        matrix=probabilities(candidate,x,classes)
        require(np.isfinite(matrix).all() and matrix.shape==values[0].shape and np.allclose(matrix.sum(1),1.,atol=1e-5),
                "Invalid added seed")
        path=root/(key+".npy")
        np.save(path,matrix,allow_pickle=False)
        save(root/(key+".json"),dict(fits=1,probability=sha(path)))
        values.append(matrix)
    return values


def build(out,i):
    root=out/"seed_stopping"/str(i)
    root.mkdir(parents=True,exist_ok=False)
    fold=get_fold(out,48,i)
    side=fold/"regular"
    src=side/"run/agent"
    old=out/"seed_replication"/str(i)
    verified=read(side/"sealed.json")
    receipt=read(old/"sealed.json")
    require(receipt["source"]==sha(side/"sealed.json") and receipt["predictions"]["always"]==sha(old/"always.npy")
            and verified["prediction"]==sha(side/"solve.csv")==sha(side/"reload.csv"),"Changed three-seed control")
    members=verified["members"]
    history={r["candidate"]:r["trial"] for r in read(src/"history.json") if r["status"]=="completed" and r["split_id"]==SPLIT}
    data_path=src/(SPLIT+".joblib")
    meta=read(src/(SPLIT+"_split.json"))
    require(sha(data_path)==meta["payload_sha256"] and meta["resource"]["shared_groups"]==0, "Unbound final training panel")
    payload=joblib.load(data_path)
    metric=MetricSpec.from_payload(payload)
    classes=np.asarray(metric.labels)
    require(metric.name=="accuracy" and len(classes)==2,"This protocol requires binary accuracy")
    y=np.asarray(payload["y_search"])
    priors, models=[],[]
    for j,m in enumerate(members):
        b=joblib.load(src/f"trial_{history[m]}.joblib")
        record=b["prediction_record"]
        require(record["evaluation_payload_sha256"]==meta["payload_sha256"],"Changed member training evidence")
        p,order=checked_score_record(record["score_record"],record,metric,m,y)
        priors.append(three(old,f"{SPLIT}_{j}",p[:,[list(order).index(c) for c in classes]]))
        models.append(b["model"])
    original=np.mean(priors,axis=0)
    bins=original[:,:,1]/original.sum(axis=2)
    uncertain=float(np.mean(np.abs(bins.mean(0)-.5)<1.64*bins.std(axis=0,ddof=1)/np.sqrt(3)))
    result=None
    if uncertain>=.01:
        added=[extend(root,f"pilot_{j}",m,payload["X_train"],payload["y_train"],payload["X_search"],classes,p)
               for j,(m,p) in enumerate(zip(models,priors,strict=True))]
        groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=payload["X_search"]))
        result=evidence(metric,y,original.mean(0),np.mean(added,axis=(0,1)),groups)
    accept=bool(result and result["accept"])
    save(root/"decision.json",dict(protocol=sha(PLAN),incumbent=sha(old/"sealed.json"),uncertain_fraction=uncertain,
                                  pilot=result,add_full_pair=accept))
    full=joblib.load(src/"refit_data.joblib")
    query_path=fold/"anonymous/queries.zip"
    require(sha(query_path)==read(fold/"inputs.json")["entry_queries"],"Changed external features")
    x,=arrays(query_path,["C_test.npy"])
    first, base, all_five=[],[],[]
    for j,m in enumerate(members):
        model=joblib.load(src/f"refit_component_{j}.joblib")["model"]
        single=probabilities(model,x,classes)
        previous=three(old,f"full_{j}",single)
        first.append(single)
        base.append(previous)
        all_five.append(extend(root,f"full_{j}",model,full["X_train"],full["y_train"],x,classes,previous))
    first=classes[np.mean(first,axis=0).argmax(1)].astype(str)
    prior=classes[np.mean(base,axis=(0,1)).argmax(1)].astype(str)
    more=classes[np.mean(all_five,axis=(0,1)).argmax(1)].astype(str)
    require(np.array_equal(first,prediction_csv(side/"solve.csv",len(x)))
            and np.array_equal(prior,np.load(old/"always.npy",allow_pickle=False)), "Wrong ordinary or three-seed control")
    stamps={}
    for name,values in (("adaptive",more if accept else prior),("always",more)):
        path=root/(name+".npy")
        np.save(path,values,allow_pickle=False)
        stamps[name]=sha(path)
    counts={k:sum(read(p)["fits"] for p in root.glob(k+"_*.json") if not p.name.endswith("_attempt.json")) for k in ("pilot","full")}
    eligible=receipt["additional_full_fits"]//2
    actual=verified["fits"]+3*eligible+counts["pilot"]+(counts["full"] if accept else 0)
    require(actual<=64 and sum(counts.values())==len(list(root.glob("*_attempt.json"))), "Unexpected marginal cost")
    save(root/"sealed.json",dict(decision=sha(root/"decision.json"),predictions=stamps,fits=counts,simulated_path_fits=actual,
                                simulated_path_assumption="ordinary solve, one probe and two original full repeats per eligible member, then the stopping policy"))
    print(i,uncertain,accept,counts,flush=True)


def grade(out):
    area=out/"seed_stopping"
    save(area/"lock.json",dict(protocol=sha(PLAN),folds={str(i):sha(area/str(i)/"sealed.json") for i in range(5)}))
    records,groups,hits=[],[],{k:[] for k in ("prior","adaptive","always")}
    for i in range(5):
        fold=get_fold(out,48,i)
        root=area/str(i)
        seed=out/"seed_replication"/str(i)
        seal=read(root/"sealed.json")
        require(sha(fold/"outer_answers.joblib")==read(fold/"inputs.json")["outer_answers"]
                and sha(seed/"always.npy")==read(seed/"sealed.json")["predictions"]["always"],"Wrong five-seed assessment packet")
        answer=joblib.load(fold/"outer_answers.joblib")
        y=answer["y"].astype(str)
        groups.extend(answer["groups"])
        values={"prior":np.load(seed/"always.npy",allow_pickle=False)}
        for k in ("adaptive","always"):
            path=root/(k+".npy")
            require(sha(path)==seal["predictions"][k],"Changed five-seed predictions")
            values[k]=np.load(path,allow_pickle=False)
        correct={}
        for k,v in values.items():
            a=v==y
            hits[k].extend(a)
            correct[k]=int(a.sum())
        records.append(dict(fold=i,rows=len(y),fits=seal["fits"],decision=read(root/"decision.json"),correct=correct,
                            simulated_path=seal["simulated_path_fits"]))
    comparisons={k:compare(np.asarray(hits[k]),np.asarray(hits["prior"]),groups) for k in ("adaptive","always")}
    delta=[(r["correct"]["adaptive"]-r["correct"]["prior"])/r["rows"] for r in records]
    result=dict(lock=sha(area/"lock.json"),folds=records,rows=len(groups),correct={k:int(np.sum(v)) for k,v in hits.items()},versus_three=comparisons,
                fits=sum(sum(r["fits"].values()) for r in records),future_cold_study=bool(comparisons["adaptive"]["gain"]>=.002
                    and sum(v>0 for v in delta)>=3 and min(delta)>=-.003),api_requests=0)
    save(area/"result.json",result)
    print(result["correct"],comparisons,result["future_cold_study"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    if args.action=="build":
        for i in range(5):
            build(args.out,i)
    else:
        grade(args.out)
