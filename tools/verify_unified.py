"""Compare unified-policy behavior to its two measured paths and reload anonymous outcomes."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from superagent.method_choice import resolved_mode
from superagent.pretrained_choice import digest
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, write


def verify_task(stage, reference, reference_arm):
    protocol=read(stage/"protocol.json")
    original=read(reference/"protocol.json")
    expect(protocol["indices"]==original["indices"] and protocol["archive_sha256"]==original["archive_sha256"],
           "Cannot compare unaligned runs")
    output=read(stage/"results.json")
    folds=[]
    for i,locations in enumerate(protocol["locations"]):
        current=Path(locations["auto"])
        earlier=Path(original["locations"][i][reference_arm])
        now=receipt(current)
        past=receipt(earlier)
        history=read(current/"trials.json")
        before=read(earlier/"trials.json")
        modes=[]
        for record in now["decisions"]:
            expect(record["mode"]=="auto" and record["resolved_mode"]==resolved_mode("auto",record["eligible"]),
                   "Wrong resolved policy")
            modes.append(record["resolved_mode"])
        changes=[]
        for n,(a,b) in enumerate(zip(history,before,strict=True),1):
            if a["candidate"]!=b["candidate"]:
                changes.append(None)
                continue
            left=joblib.load(current/f"trial_{n}.joblib")["search_predictions"]
            right=joblib.load(earlier/f"trial_{n}.joblib")["search_predictions"]
            expect(left.shape==right.shape,"Search row counts differ")
            changes.append(int(np.sum(left!=right)))
        expect(changes[:5]==[0]*5,"The ordinary five-trial baseline changed")
        n=protocol["fold_numbers"][i]-1
        held=read(stage/f"fold_{n}"/"outer_predictions.json")
        old=read(reference/f"fold_{n}"/"outer_predictions.json")
        expect(held["rows"]==old["rows"] and held["truth"]==old["truth"],"Outer rows changed")
        changes_outer=int(np.sum(np.asarray(held["auto"])!=old[reference_arm]))
        folds.append(dict(resolved_modes=modes,same_action_sequence=all(c is not None for c in changes),
                          changed_search_predictions=changes,same_final_candidate=now["winner"]==past["winner"],
                          changed_final_predictions=changes_outer,rows=len(held["rows"]),
                          correct=int(np.sum(np.asarray(held["auto"])==held["truth"]))))
    expect(sum(f["correct"] for f in folds)==output["correct"]["auto"],"Published unified score changed")
    return dict(rows=output["rows"],correct=output["correct"]["auto"],reference_arm=reference_arm,folds=folds,
                protocol_sha256=digest(stage/"protocol.json"),reference_protocol_sha256=digest(reference/"protocol.json"))


def smoke_resources(root):
    fits=api_calls=input_tokens=batches=0
    for file in root.glob("*/run/summary.json"):
        summary=read(file)
        fits+=summary["classifier_fit_accounting"]["completed_classifier_fits"]
        api_calls+=summary["llm_calls"]
        expect(summary["actual_cost_rmb"]==0,"Unexpected service debit on an anonymous fixture")
    for file in root.rglob("downstream_usage.jsonl"):
        for line in file.read_text().splitlines():
            event=json.loads(line)
            expect(event["remote_calls"]==event["api_rmb"]==event["output_tokens"]==0,"Unexpected feature usage")
            input_tokens+=event["input_tokens"]
            batches+=1
    return dict(fits=fits,api_calls=api_calls,minilm_input_tokens=input_tokens,minilm_batches=batches,api_rmb=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("task1","task2","task1-reference","task2-reference","smoke","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--additional-smoke",type=Path,nargs="*",default=[])
    args=parser.parse_args()
    p1=read(args.task1/"protocol.json")
    p2=read(args.task2/"protocol.json")
    expect(p1["source"]==p2["source"] and p1["options"]==p2["options"],"Different frozen implementations or budgets")
    repo=Path(__file__).resolve().parents[1]
    for file,sha in p1["source"].items():
        expect(digest(repo/file)==sha,f"Source no longer matches the evaluated version: {file}")
    runs=dict(task1=verify_task(args.task1,args.task1_reference,"paired"),
              task2=verify_task(args.task2,args.task2_reference,"text-encoder"))
    cases={}
    for label,smoke in read(args.smoke/"smoke_summary.json").items():
        folder=args.smoke/label
        expect(read(folder/"run"/"method_policy.json")["mode"]=="auto","Smoke fixture used a different policy")
        manifest=read(folder/"prediction_manifest.json")
        expect(digest(folder/"predictions.csv")==manifest["prediction_sha256"] and not manifest["test_labels_read"],
               "Anonymous output changed or accessed a label")
        with (folder/"predictions.csv").open() as f:
            ids=[int(row["row_id"]) for row in csv.DictReader(f)]
        expect(ids==list(range(smoke["predictions"])),"Anonymous output row order changed")
        case={k:smoke[k] for k in ("kind","metric","trials","winner","predictions","stop","api_calls")}
        case["minilm_training_input_tokens"]=smoke.get("minilm_input_tokens",0)
        case["prediction_feature_usage"]=manifest.get("prediction_feature_usage")
        if (folder/"relocated"/"checks.json").exists():
            moved=read(folder/"relocated"/"checks.json")
            expect(moved["byte_identical"] and moved["cached"]["input_tokens"]==0 and moved["first"]["input_tokens"]>0,
                   "Relocated output did not pass both inference checks")
            expect(digest(folder/"relocated"/"predictions.csv")==manifest["prediction_sha256"],"Moved output differs")
            case["relocated"]=moved
        cases[label]=case
    expect(len(cases)==6,"Expected six different anonymous fixtures")
    accounts={root.name:smoke_resources(root) for root in [args.smoke,*args.additional_smoke]}
    write(args.output,dict(tasks=runs,anonymous=cases,anonymous_resources=accounts,source=p1["source"],options=p1["options"]))
    print({k:(v["correct"],v["rows"],sum(f["changed_final_predictions"] for f in v["folds"])) for k,v in runs.items()},flush=True)


if __name__=="__main__":main()
