"""Prepare isolated training roles and execute the original agent at matched fit ceilings."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_planner_self_review import source_identity
from scripts.run_task1_profile_screen import load_split
from scripts.unified_released_study import ROOT,env,make_zip,read,save,sha,require
from superagent.splits import sample_groups


PLAN=ROOT/"research/task1_feedback_extension_1127/training_protocol.json"
ROLES=("nomination","confirmation","audit")


def seeded(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8],16)


def key(indices):
    return hashlib.sha256(np.asarray(indices,dtype="<i8").tobytes()).hexdigest()


def npy(rows):
    sink=io.BytesIO()
    np.save(sink,rows,allow_pickle=False)
    return sink.getvalue()


def prepare(archive,out):
    out.mkdir(parents=True,exist_ok=True)
    x,y,access=load_split(archive,"train")
    require(len(y)==len(x),"Mismatched training rows")
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    seed=seeded("AIAA recurring table extension outer v1")
    inner_seed=seeded("AIAA recurring table extension inner v1")
    dev,external=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(x,y,groups))
    ix=[dev[h] for _,h in StratifiedGroupKFold(8,shuffle=True,random_state=inner_seed).split(x[dev],y[dev],groups[dev])]
    roles=dict(zip(ROLES,ix[:3],strict=True),fit=np.sort(np.concatenate(ix[3:])),external=external)
    require(sum(len(set(groups[r])) for r in roles.values())==len(set(groups))
            and len(np.unique(np.concatenate(list(roles.values()))))==len(x),"Overlapping evaluation roles")
    public=out/"anonymous"
    public.mkdir(exist_ok=True)
    inputs=public/"training.zip"
    query=public/"queries.zip"
    if not inputs.exists():
        make_zip(inputs,[("C_train.npy",npy(x[dev].astype(str))),("y_train.npy",npy(y[dev])),
              ("Readme.md",b"Binary categorical classification. The primary metric is accuracy.")])
        make_zip(query,[("C_test.npy",npy(x[external].astype(str)))])
        joblib.dump(dict(y=y[external],groups=groups[external]),out/"outer_answers.joblib")
        joblib.dump(dict(x=x[roles["fit"]],y=y[roles["fit"]],groups=groups[roles["fit"]]),out/"fitting.joblib")
        for role in ROLES:
            joblib.dump(x[roles[role]],out/(role+"_features.joblib"))
            joblib.dump(y[roles[role]],out/(role+"_answers.joblib"))
    info=dict(protocol=sha(PLAN),training_members=access,seed=seed,inner_seed=inner_seed,
              development=len(dev),rows=len(y),features=x.shape[1],roles={k:dict(rows=len(v),groups=len(set(groups[v])),
              digest=key(v)) for k,v in roles.items()},partitions_disjoint=True,
              entry_inputs=sha(inputs),entry_queries=sha(query),outer_answers=sha(out/"outer_answers.joblib"),
              packets={p.name:sha(p) for p in sorted(out.glob("*.joblib"))},official_test_opened=False,validation_opened=False)
    save(out/"inputs.json",info)
    print({k:v["rows"] for k,v in info["roles"].items()},flush=True)


def entry(source, out, arm, cap, table_weight=None, optional_runtime=None, feedback_mode=None):
    data=read(out/"inputs.json")
    inputs=out/"anonymous/training.zip";query=out/"anonymous/queries.zip"
    require(sha(inputs)==data["entry_inputs"] and sha(query)==data["entry_queries"],"Anonymous packet changed")
    folder=out/arm;folder.mkdir(exist_ok=False)
    settings=env(folder/"unused",folder/"physical.jsonl")
    for k in list(settings):
        if k.startswith("SUPERAGENT_"):
            settings.pop(k)
    modules = [source/"scripts/unified_acceptance", source]
    if optional_runtime:
        modules.append(optional_runtime)
    resources = {}
    if table_weight:
        from superagent.table_foundation import SHA256, WEIGHT_ENV
        require(sha(table_weight) == SHA256, "Unverified optional resource")
        settings[WEIGHT_ENV] = str(table_weight)
        resources["tabicl_v2"] = SHA256
    settings.update(PYTHONPATH=os.pathsep.join(map(str, modules)),
                    UNIFIED_RAW_INPUT=str(inputs),UNIFIED_RAW_LOG=str(folder/"opened.jsonl"),
                    UNIFIED_RAW_MEMBERS=json.dumps(["C_train.npy","y_train.npy","Readme.md"]),SUPERAGENT_OFFLINE="1")
    save(folder/"launch.json",dict(inputs=sha(out/"inputs.json"),source=source_identity(source),cap=cap,seconds=1800,
                                   method_hint=None, optional_resources=resources, feedback_mode=feedback_mode))
    mode_flags=["--feedback-reflection",feedback_mode] if feedback_mode else []
    for phase,args in [("solve",["solve",inputs,"--prediction-input",query,"--fit-limit",cap,"--time-limit",1800,*mode_flags]),
                       ("reload",["predict",query])]:
        with (folder/(phase+".log")).open("x") as stream:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,args),"--run-dir",str(folder/"run"),
                            "--output",str(folder/(phase+".csv"))],cwd=source,env=settings,stdin=subprocess.DEVNULL,
                            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=2000)
    native=folder/"run/agent"
    done=read(native/"run_outcome.json")
    chosen=read(native/"final_selection.json")
    count=len((folder/"physical.jsonl").read_text().splitlines())
    files=sorted({json.loads(s) for s in (folder/"opened.jsonl").read_text().splitlines()})
    result=sha(folder/"solve.csv")
    require(done["status"]=="completed" and done["new_api_requests"]==0 and count==done["attempted_classifier_fits"]<=cap
            and chosen["model_sha256"]==sha(native/chosen["model_file"]) and result==sha(folder/"reload.csv"),
            "Incomplete run, physical accounting or reload")
    with (folder/"solve.csv").open() as file:
        predictions=list(csv.DictReader(file))
    require([r["row_id"] for r in predictions]==list(map(str,range(data["roles"]["external"]["rows"]))),"Unordered prediction")
    seal=dict(launch=sha(folder/"launch.json"),members=done["members"],model=chosen["model_sha256"],
              prediction=result,rows=len(predictions),fits=count,status=done["status"],seconds=done["wall_seconds"],
              independent_reload=True,opened=files,model_api_requests=0,score=None)
    save(folder/"sealed.json",seal)
    print(arm,seal,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","entry"])
    p.add_argument("--archive",type=lambda s:Path(s).resolve())
    p.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    p.add_argument("--source",type=lambda s:Path(s).resolve())
    p.add_argument("--arm")
    p.add_argument("--cap",type=int,choices=(10,32))
    p.add_argument("--table-weight", type=lambda s: Path(s).resolve())
    p.add_argument("--optional-runtime", type=lambda s: Path(s).resolve())
    p.add_argument("--feedback-mode",choices=["fusion-holdout","near-full-confirmation"])
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.archive,args.out)
    else:
        entry(args.source,args.out,args.arm,args.cap,args.table_weight,args.optional_runtime,args.feedback_mode)
