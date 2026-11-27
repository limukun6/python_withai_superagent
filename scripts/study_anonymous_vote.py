"""Cold-run an already open image development packet without sending the planner a method hint."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from scripts.unified_released_study import ROOT,env,read,require,save,sha


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("development","weights","output"):
        p.add_argument("--"+n,type=lambda v:Path(v).resolve(),required=True)
    args=p.parse_args()
    folder=args.development/"v83a"
    entry=read(args.development/"registered.json")["cases"][folder.name]
    train,query=folder/"task.zip",folder/"query.zip"
    require(sha(train)==entry["train_sha256"] and sha(query)==entry["query_sha256"],"Registered inputs differ")
    args.output.mkdir(parents=True,exist_ok=False)
    physical=args.output/"physical.jsonl"
    settings=env(args.weights,physical)
    for key in ("SUPERAGENT_PLANNER_KEY_FILE","SUPERAGENT_PLANNER_LEDGER","SUPERAGENT_TASK_BUDGET_ID",
                "SUPERAGENT_RESEARCH_REFLECTION_OUTBOX"):
        settings.pop(key,None)
    run=args.output/"run"
    predictions=args.output/"delivery/predictions.csv"
    save(args.output/"protocol.json",dict(protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/image_protocol.json"),
         train_sha256=sha(train),query_sha256=sha(query),external_answers_opened=False))
    with (args.output/"solve.log").open("x") as log:
        subprocess.run([sys.executable,"-m","unified_agent","solve",str(train),"--feedback-reflection","matched-finalists",
                        "--fit-limit","16","--time-limit","600","--run-dir",str(run),"--prediction-input",str(query),
                        "--output",str(predictions)],cwd=ROOT,env=settings,stdout=log,stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL,timeout=675,check=True)
    agent=run/"agent"
    outcome=read(agent/"run_outcome.json")
    fits=len(physical.read_text().splitlines())
    with predictions.open() as stream:
        rows=list(csv.DictReader(stream))
    require(outcome["status"]=="completed" and outcome["new_api_requests"]==0 and
            fits==outcome["attempted_classifier_fits"]<=16 and len(rows)==entry["held_rows"] and
            read(agent/"final_selection.json")["refit"],"Incomplete anonymous run")
    record=dict(members=outcome["members"],actual_fits=fits,new_model_requests=0,predicted_rows=len(rows),
                training_rows=outcome["training_rows"],external_answers_opened=False,predictions_sha256=sha(predictions))
    if (agent/"finalist_result.json").is_file():
        record["finalists"]=read(agent/"finalist_result.json")
    save(args.output/"results.json",record)
    print(record,flush=True)
