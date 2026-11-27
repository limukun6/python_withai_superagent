"""Retain the actual deadline checkpoint if the ordinary unexpanded control is partial."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.unified_released_study import ROOT,env,read,require,save,sha
from scripts.verify_unified_raw import MEMBERS


def seal(folder, features, weights):
    agent=folder/"run/agent"
    result=read(agent/"run_outcome.json")
    choice=read(agent/"final_selection.json")
    require(result["status"]=="partial" and result["model_available"] and result["new_api_requests"]==0
            and not result["formal_test_accessed"],"Not a valid time-limited partial checkpoint")
    opened=sorted({json.loads(s) for s in (folder/"opened.jsonl").read_text().splitlines()})
    require(opened==sorted(MEMBERS["titles"]) and sha(agent/choice["model_file"])==choice["model_sha256"],
            "Training scope or checkpoint mismatch")
    fits=folder/"physical.jsonl"
    initial=len(fits.read_text().splitlines())
    settings=env(weights,fits)
    for key in ("SUPERAGENT_PLANNER_KEY_FILE","SUPERAGENT_PLANNER_LEDGER","SUPERAGENT_TASK_BUDGET_ID","SUPERAGENT_PAIRED_FEATURE_CACHE"):
        settings.pop(key,None)
    predictions=folder/"delivery/predictions.csv"
    replay=folder/"deadline-replay/predictions.csv"
    with (folder/"deadline-replay.log").open("x") as out:
        subprocess.run([sys.executable,"-m","unified_agent","predict",str(features),"--run-dir",str(folder/"run"),
                       "--output",str(replay)],cwd=ROOT,env=settings,stdout=out,stderr=subprocess.STDOUT,
                       check=True,stdin=subprocess.DEVNULL,timeout=240)
    require(sha(predictions)==sha(replay) and len(fits.read_text().splitlines())==initial and
            initial==result["attempted_classifier_fits"],"Partial replay or cost mismatch")
    save(folder/"sealed.json",dict(status="partial",reason=result.get("error"),prediction_sha256=sha(predictions),
         model_sha256=choice["model_sha256"],members=[choice["single_candidate"]] if not choice["ensemble_components"] else choice["ensemble_components"],
         fits=initial,completed_fits=result["completed_classifier_fits"],unknown_failed_attempts=result["unknown_failed_attempts"],
         rows=choice["training_rows"],refit=choice["refit"],independent_reload=True,api=0,opened=opened))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("folder","features","weights"):
        p.add_argument("--"+n,type=lambda s:Path(s).resolve(),required=True)
    args=p.parse_args()
    seal(args.folder,args.features,args.weights)
