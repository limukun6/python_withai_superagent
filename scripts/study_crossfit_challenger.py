"""Prepare anonymous grouped tasks and exercise the same opt-in course CLI on each."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.prepare_conditional_handoff import partition
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha, text_packet
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from unified_agent import audit_partition

PLAN = ROOT/"docs/batches/crossfit_challenger_plan_20261124_zh.md"


def call(args, log):
    env = dict(os.environ, AB_B1_OFFLINE="1", SUPERAGENT_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               PYTHONPATH=os.pathsep.join(map(str,(ROOT/"scripts/ab_acceptance_guard",ROOT))))
    with log.open("x") as stream:
        subprocess.run([sys.executable,"-m","unified_agent",*map(str,args)],cwd=ROOT,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def native_train(folder, weights, cache):
    run = folder/"run"
    if not (run/"challenger_completion.json").exists():
        require(not run.exists() and not (folder/"training.log").exists(),"Partial path already exists; inspect its fit receipts")
        call(["train",folder/"train.zip","--run-dir",run,"--strategy","crossfit-challenger","--weights",weights,
              "--feature-bank",cache],folder/"training.log")
    completion = read(run/"challenger_completion.json")
    require(completion["classifier_fits"]<=9 and completion["model_api_calls"]==0 and
            read(run/"challenger_protocol.json")["package"]==sha(folder/"train.zip")
            and completion["model_sha256"]==sha(run/"agent/selected_model.joblib"),"Unbound model or budget")
    return completion


def native_predict(folder,weights,cache=None):
    output = folder/("cached" if cache is not None else "direct")/"predictions.csv"
    if not output.exists():
        start = time.monotonic()
        args = ["predict",folder/"features.zip","--run-dir",folder/"run","--weights",weights,"--output",output]
        if cache is not None:
            args.extend(("--feature-bank",cache))
        call(args,folder/(output.parent.name+".log"))
        save(output.parent/"elapsed.json",{"seconds":round(time.monotonic()-start,4)})
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["row_id"]) for r in rows]==list(range(len(rows))),"Mismatched native row order")
    return np.array([int(r["prediction"]) for r in rows])


def smoke(out,weights,cache):
    out.mkdir(parents=True,exist_ok=False)
    rows = []
    for i in range(180):
        left = f"{'early' if i%2 else 'late'} meeting brings reports from district {i}"
        right = f"what {'residents' if i%3 else 'visitors'} say about the meeting in district {i}"
        rows.append((left,right,int(((i%2)==0) ^ ((i%7)==0))))
    text_packet(out/"train.zip",rows[:162],True)
    text_packet(out/"features.zip",[r[:2] for r in rows[162:]],False)
    receipt = native_train(out,weights,cache)
    first,second = native_predict(out,weights,cache),native_predict(out,weights)
    require(len(first)==18 and np.array_equal(first,second),"Anonymous prediction export disagrees")
    save(out/"smoke.json",{"rows":18,"classifier_fits":receipt["classifier_fits"],"decision":receipt["strategy"],
         "model":receipt["model_sha256"],"prediction":sha(out/"direct/predictions.csv"),
         "matches":sha(out/"direct/predictions.csv")==sha(out/"cached/predictions.csv"),"model_api_calls":0})
    print(read(out/"smoke.json"),flush=True)


def study_splits(archive):
    task = load_training_task(archive)
    audit,_ = audit_partition(task)
    require(audit is not None,"Expected historical group reserve")
    main,reserve,_ = audit
    require(np.array_equal(reserve,read(EVIDENCE/"title_reserve_ids.json")),"Historical reserve differs")
    groups = sample_groups(task,"shared_text")
    splits = []
    for roles in partition(groups,reserve):
        held = roles["outer"]
        fit = np.setdiff1d(main,held)
        require(not set(groups[fit]) & set(groups[held]) and not set(groups[fit]) & set(groups[reserve]),"Overlapping model/evaluation groups")
        splits.append((fit,held))
    return task,groups,splits


def prepare(archive, out):
    out.mkdir(parents=True,exist_ok=False)
    task,groups,splits = study_splits(archive)
    protocol = {}
    for i,(fit,held) in enumerate(splits):
        dest = out/f"fold_{i}"
        dest.mkdir()
        text_packet(dest/"train.zip",[(*task.X_train[j],int(task.y_train[j])) for j in fit],True)
        text_packet(dest/"features.zip",[task.X_train[j] for j in held],False)
        protocol[str(i)] = {k:{"rows":len(ix),"ids":fingerprint(ix.tolist()),"groups":len(set(groups[ix]))}
                            for k,ix in (("training",fit),("outer",held))}
        protocol[str(i)].update(train_packet=sha(dest/"train.zip"),features_packet=sha(dest/"features.zip"))
    save(out/"protocol.json",dict(plan=sha(PLAN),folds=protocol,maximum_fits_per_path=9,model_api_limit=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","prepare","train"))
    p.add_argument("--archive",type=lambda s:Path(s).resolve())
    p.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    p.add_argument("--weights",type=lambda s:Path(s).resolve())
    p.add_argument("--cache",type=lambda s:Path(s).resolve())
    p.add_argument("--fold",type=int,choices=range(5))
    a=p.parse_args()
    if a.stage=="prepare":
        prepare(a.archive,a.out)
    elif a.stage=="smoke":
        smoke(a.out,a.weights,a.cache)
    else:
        protocol=read(a.out/"protocol.json")
        require(protocol["plan"]==sha(PLAN),"Plan changed")
        for i in (range(5) if a.fold is None else [a.fold]):
            folder=a.out/f"fold_{i}"
            require(sha(folder/"train.zip")==protocol["folds"][str(i)]["train_packet"],"Training input changed")
            completion=native_train(folder,a.weights,a.cache)
            print("fold",i,completion["classifier_fits"],completion["strategy"],flush=True)
