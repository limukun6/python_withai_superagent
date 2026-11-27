"""Run a complete course solve with a generic asset inventory; evaluate only separately sealed runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, env, read, require, save, sha, title_rows
from scripts.verify_unified_raw import MEMBERS
from superagent.ab_sentence import ALTERNATIVES
from superagent.paired_encoder import VARIANTS


PROTOCOL=ROOT/"research/paired_encoder_search_1127/raw_protocol.json"


def solve(args):
    folder=args.output/args.arm
    folder.mkdir(parents=True,exist_ok=False)
    study_plan=PROTOCOL.with_name("race_protocol.json") if args.arm=="race" else PROTOCOL
    policy=read(study_plan)
    course=archive(args.course,"titles")
    raw_files=read(ROOT/"research/b_task2_handoff_1127/protocol.json")["input"]
    require(sha(course)==raw_files["raw"] and sha(args.features)==raw_files["features"],"Unknown input")
    physical=folder/"physical.jsonl"
    opened=folder/"opened.jsonl"
    settings=env(args.weights,physical)
    for name in ("SUPERAGENT_PLANNER_KEY_FILE","SUPERAGENT_PLANNER_LEDGER","SUPERAGENT_TASK_BUDGET_ID",
                 "SUPERAGENT_RESEARCH_REFLECTION_OUTBOX","SUPERAGENT_PAIRED_FEATURE_CACHE"):
        settings.pop(name,None)
    for name in VARIANTS:
        spec=ALTERNATIVES[name]
        settings.pop(spec["env"],None)
        if args.arm != "original":
            settings[spec["env"]]=str(args.weights/spec["model"].split("/")[-1])
    if args.arm != "original":
        settings["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(args.cache)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/"scripts/unified_acceptance",ROOT))),
                    UNIFIED_RAW_INPUT=str(course),UNIFIED_RAW_MEMBERS=json.dumps(MEMBERS["titles"]),UNIFIED_RAW_LOG=str(opened))
    def execute(arguments,log,environment):
        with (folder/log).open("x") as stream:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,arguments)],cwd=ROOT,env=environment,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1400)
    run=folder/"run"
    result=folder/"delivery/predictions.csv"
    save(folder/"protocol.json",dict(runtime_sha256=source_identity(ROOT),plan_sha256=sha(study_plan),arm=args.arm,
         candidate_hint=None,fit_limit=policy["fit_limit_each_including_refit"],raw_sha256=sha(course),features_sha256=sha(args.features)))
    review=["--feedback-reflection","matched-finalists"] if args.arm=="race" else []
    execute(["solve",course,*review,"--fit-limit",policy["fit_limit_each_including_refit"],"--time-limit",policy["wall_seconds_each"],
             "--run-dir",run,"--prediction-input",args.features,"--output",result],"solve.log",settings)
    agent=run/"agent"
    outcome,chosen=read(agent/"run_outcome.json"),read(agent/"final_selection.json")
    actual=len(physical.read_text().splitlines())
    require(outcome["status"]=="completed" and outcome["attempted_classifier_fits"]==actual<=policy["fit_limit_each_including_refit"]
            and outcome["new_api_requests"]==0 and not outcome["formal_test_accessed"],"Search or usage mismatch")
    members=sorted({json.loads(line) for line in opened.read_text().splitlines()})
    require(members==sorted(MEMBERS["titles"]) and chosen["refit"] and sha(agent/chosen["model_file"])==chosen["model_sha256"],
            "Model or training boundaries changed")
    replay=folder/"replay/predictions.csv"
    settings.pop("SUPERAGENT_PAIRED_FEATURE_CACHE",None)
    execute(["predict",args.features,"--run-dir",run,"--output",replay],"reload.log",settings)
    require(sha(result)==sha(replay) and len(physical.read_text().splitlines())==actual,"Uncached reload differs")
    save(folder/"sealed.json",dict(prediction_sha256=sha(result),model_sha256=chosen["model_sha256"],fits=actual,
         members=read(agent/"decisions.json")[-1]["choice"]["members"],opened=members, independent_reload=True,
         rows=chosen["training_rows"],api=outcome["new_api_requests"]))
    print(args.arm,read(folder/"sealed.json"),flush=True)


def score(args):
    folders=[args.reference,*[args.output/name for name in args.sealed_arms]]
    require(len(args.sealed_arms)==len(set(args.sealed_arms)) and set(args.sealed_arms)=={"original","expanded","race"},
            "The three registered autonomous cold runs must be sealed before evaluation")
    predictions=[]
    for folder in folders:
        saved=read(folder/"sealed.json")
        path=folder/"delivery/predictions.csv"
        require(saved["independent_reload"] and saved["prediction_sha256"]==sha(path),"Unsigned predictions")
        with path.open(encoding="utf-8",newline="") as file:
            rows=list(csv.DictReader(file))
        require(len(rows)==940 and [x["row_id"] for x in rows]==[str(i) for i in range(940)],"Row mismatch")
        predictions.append(np.array([int(x["prediction"]) for x in rows]))
    save(args.output/"evaluation_lock.json",dict(arms=args.sealed_arms,baseline=sha(folders[0]/"sealed.json"),
         seals={folder.name:sha(folder/"sealed.json") for folder in folders[1:]}))
    truth=np.array([r[2] for r in title_rows(archive(args.course,"titles"),"val",labels=True)])
    base=predictions[0]==truth
    report={}
    for name,labels in zip(["delivered_10_fits",*args.sealed_arms],predictions):
        right=labels==truth
        report[name]=dict(correct=int(sum(right)),rows=len(right),accuracy=float(right.mean()),
                          fixed=int(sum(right & ~base)),lost=int(sum(base & ~right)))
    names=["delivered_10_fits",*args.sealed_arms]
    paired=[]
    for i,first in enumerate(predictions):
        for j,second in enumerate(predictions[i+1:],i+1):
            before,after=first==truth,second==truth
            paired.append(dict(reference=names[i],candidate=names[j],fixed=int(sum(after & ~before)),
                               lost=int(sum(before & ~after)),both_correct=int(sum(before & after)),
                               both_incorrect=int(sum(~before & ~after))))
    report.update(official_test_opened=False,paired_comparisons=paired)
    save(args.output/"validation.json",report)
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["solve","score"])
    parser.add_argument("--arm",choices=["original","expanded","race"])
    parser.add_argument("--sealed-arms",nargs="+")
    for name in ("course","features","weights","cache","output","reference"):
        parser.add_argument("--"+name,type=lambda s:Path(s).resolve(),required=name in ("course","output"))
    args=parser.parse_args()
    (solve if args.mode=="solve" else score)(args)
