"""Reconstruct an unconstrained-time, equal-fit control from exact old-runtime observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT,archive,env,read,require,save,sha
from scripts.verify_unified_raw import MEMBERS
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_run import Executor,registered_capabilities
from superagent.autonomous_run import freeze_model,planner_observation
from superagent.contracts import TaskSpec
from superagent.events import RunLogger
from superagent.experiment_policy import actions
from superagent.metrics import MetricSpec
from superagent.open_methods import MethodSpace
from superagent.semantic_contract import resolve_training_task


def run(args):
    original=args.raw/"original"
    origin=read(original/"protocol.json")
    require(read(original/"sealed.json").get("status")=="partial" and source_identity(ROOT)==origin["runtime_sha256"],
            "Invoke with the original training source on the import path and a sealed partial control")
    archive_path=archive(args.course,"titles")
    require(sha(archive_path)==origin["raw_sha256"] and sha(args.features)==origin["features_sha256"],"Unregistered input")
    folder=args.raw/"original_compute_matched"
    agent=folder/"run/agent"
    agent.mkdir(parents=True,exist_ok=False)
    physical=folder/"new_physical.jsonl"
    environment=env(args.weights,physical)
    for key in ("SUPERAGENT_GTE_SMALL_WEIGHTS","SUPERAGENT_ELECTRA_SMALL_WEIGHTS","SUPERAGENT_PLANNER_KEY_FILE",
                "SUPERAGENT_PLANNER_LEDGER","SUPERAGENT_TASK_BUDGET_ID","SUPERAGENT_PAIRED_FEATURE_CACHE"):
        os.environ.pop(key,None)
        environment.pop(key,None)
    os.environ.update(environment)
    os.environ[PREFERENCE_CATALOG_ENV]="all-installed"
    original_agent=original/"run/agent"
    task=resolve_training_task(archive_path,mode="fallback",prediction_package=args.features)
    contract=TaskSpec.from_task(task).record()
    metric=MetricSpec.for_task(task)
    require(contract==read(original_agent/"task_contract.json"),"Original task contract changed")
    split_names=("primary","confirmation_1","confirmation_2")
    payloads={f:original_agent/(f+".joblib") for f in split_names}
    records={f:dict(sha256=sha(v),labels=joblib.load(v)["y_search"]) for f,v in payloads.items()}
    observed={f:{} for f in split_names}
    profile=read(original_agent/"training_profile.json")
    registry=MethodSpace(contract,profile,agent)
    static=registered_capabilities(contract,len(task.y_train))
    log=RunLogger(agent)
    executor=Executor(agent,{},metric,time.monotonic(),2400.,300.,20,log,restrictions=contract.get("method_restrictions",()))
    original_steps=read(original_agent/"decisions.json")
    cold_history=read(original_agent/"history.json")
    prefix_length=next((i for i,r in enumerate(cold_history) if r["status"]!="completed"),len(cold_history))
    completed_prefix=cold_history[:prefix_length]
    donors={}
    for label in ("original","expanded"):
        donor=args.raw/label
        require(read(donor/"protocol.json")["runtime_sha256"]==origin["runtime_sha256"],"Incompatible donor runtime")
        for row in read(donor/"run/agent/history.json"):
            if row["status"]=="completed":
                key=row["split_id"],row["candidate"]
                donors.setdefault(key,(label,donor/"run/agent"/f"trial_{row['trial']}.joblib"))
    history=[]
    sequence=[]
    while True:
        caps,_=registry.refresh(history,static)
        executor.caps=caps
        remaining=20-executor.accounting()["attempted_classifier_fits"]
        selected=actions(caps,profile,history,observed,metric,list(split_names),remaining)[0]
        i=len(history)
        if i<len(completed_prefix):
            old=original_steps[i]["choice"]
            require(all(selected[k]==old[k] for k in ("action","split","members")),"Cold-prefix divergence")
        if selected["action"]=="stop":
            break
        name=selected["members"][0]
        fold=selected["split"]
        if (fold,name) in donors:
            origin_name,path=donors[fold,name]
            bundle=joblib.load(path)
            fact=bundle["metrics"]
            require(fact["resolved_configuration"]["id"]==name and fact["resolved_configuration"]["params"]==caps[name]["params"]
                    and fact["classifier_fits"]==caps[name]["classifier_fits"],"Donor configuration or physical cost differs")
            executor.records.append(dict(attempted_classifier_fits=fact["classifier_fits"],completed_classifier_fits=fact["classifier_fits"],
                                         source=origin_name,artifact_sha256=sha(path)))
        else:
            origin_name="new"
            path=agent/f"trial_{i+1}.joblib"
            executor.worker(payloads[fold],name,path,300,reserve=selected["evidence"]["reserved_refits"])
            bundle=joblib.load(path)
            fact=bundle["metrics"]
        observed[fold][name]=planner_observation(bundle,fact,records[fold],metric)
        history.append(dict(trial=i+1,candidate=name,split_id=fold,status="completed",primary_metric=metric.name,
                            train_score=fact["train_score"],search_score=fact["search_score"],search_accuracy=fact["search_accuracy"]))
        sequence.append(dict(step=i+1,action=selected["action"],members=[name],fold=fold,source=origin_name))
        log.write_json("logical_path.json",sequence)
    final=freeze_model(task,contract,metric,selected,executor,agent,log,300.)
    log.write_json("run_manifest.json",read(original_agent/"run_manifest.json"))
    predictions=[]
    for name in ("delivery","replay"):
        result=folder/name/"predictions.csv"
        with (folder/(name+".log")).open("x") as output:
            subprocess.run([sys.executable,"-m","unified_agent","predict",str(args.features),"--run-dir",str(folder/"run"),
                           "--output",str(result)],cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT,
                           check=True,stdin=subprocess.DEVNULL,timeout=300)
        predictions.append(sha(result))
    logical=executor.accounting()["attempted_classifier_fits"]
    new=len(physical.read_text().splitlines())
    reused=sum(caps[s["members"][0]]["classifier_fits"] for s in sequence if s["source"]!="new")
    require(predictions[0]==predictions[1] and new+reused==logical<=20,"Counterfactual accounting or predictions changed")
    save(folder/"sealed.json",dict(status="compute_matched_replay",members=selected["members"],model_sha256=final["model_sha256"],
         prediction_sha256=predictions[0],independent_reload=True,rows=final["training_rows"],fits=logical,
         reused_fits=reused,additional_physical_fits=new,shared_prefix_decisions=len(completed_prefix),
         study_plan_sha256=sha(args.protocol),api=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("raw","course","features","weights","protocol"):
        p.add_argument("--"+n,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
