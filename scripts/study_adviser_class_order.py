"""Replay an adviser election under both class orders and verify a real reordered task."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, make_zip, read, require, sha
from superagent.analogy_review import blend_counts, incumbent_evidence, nominate
from superagent.autonomous_run import planner_observation
from superagent.metrics import MetricSpec


PLAN = ROOT/"research/paired_encoder_search_1127/adviser_class_order_protocol.json"


def deployed_share(selection, candidate):
    if candidate is None:
        return 0.0
    members = selection.get("ensemble_components", [])
    if candidate in members:
        return selection["weights"][members.index(candidate)]
    return float(selection.get("single_candidate") == candidate)


def replay(root, out):
    agent = root/"analogy-portfolio-raw/portfolio_revised/run/agent"
    report = read(agent/"analogy_review.json")
    history = read(agent/"history.json")
    receipts = {}
    for fold in ("primary","confirmation_1","confirmation_2"):
        payload = agent/(fold+".joblib")
        truth = np.asarray(joblib.load(payload)["y_search"])
        digest = sha(payload)
        require(digest == read(agent/(fold+"_split.json"))["payload_sha256"], "Mismatched training panel")
        metric = MetricSpec("accuracy",(0,1))
        def observation(path):
            bundle = joblib.load(path)
            return planner_observation(bundle, bundle["metrics"], dict(sha256=digest, labels=truth), metric)
        fitted = {r["candidate"]:r["trial"] for r in history if r["status"] == "completed" and r["split_id"] == fold}
        records = [observation(agent/f"trial_{fitted[m]}.joblib") for m in report["incumbent_members"]]
        anchor = incumbent_evidence(records, report["incumbent_weights"], metric)
        name = report["candidate"]
        vote = observation(agent/f"analogy_{fold}_{name}.joblib")
        for classes in ((0,1),(1,0)):
            a = dict(anchor, probabilities=np.asarray(anchor["probabilities"])[:,list(classes)])
            v = dict(vote, probabilities=np.asarray(vote["probabilities"])[:,list(classes)])
            if fold == "primary":
                decision = nominate(a, v, truth, classes=classes)
                require(decision == report["transparent_nomination"], "Nomination changes when columns are relabelled")
                counts = decision["scores"]
            else:
                counts = blend_counts(a, v, truth, [0,report["locked_weight"]], classes=classes)
                actual = next(row for row in report["allocations"] if row["fold"] == fold)
                require(counts == [actual["incumbent"],actual["candidate"]], "Confirmation changed under class reorder")
            receipts.setdefault(fold, {})[str(classes)] = counts
    lock(out/"training_replay.json", dict(protocol=sha(PLAN), review=sha(agent/"analogy_review.json"),
                                          panels=receipts, fits=0, api_requests=0, outer_labels_opened=False))


def solve(root, source, out):
    packet, query = source/"anonymous.zip", source/"features.zip"
    source_record = read(source/"protocol.json")
    require(sha(packet) == source_record["train_packet"] and sha(query) == source_record["features"], "Movie bytes differ")
    with ZipFile(packet) as z:
        entries = {n:z.read(n) for n in z.namelist()}
    task = json.loads(entries["manifest.json"])
    require(task["metric"] == "accuracy" and "positive_label" not in task, "Expected ordinary movie contract")
    task["positive_label"] = 0
    entries["manifest.json"] = json.dumps(task, ensure_ascii=False).encode()
    new_packet = out/"anonymous.zip"
    make_zip(new_packet, entries.items())
    with ZipFile(packet) as old, ZipFile(new_packet) as changed:
        require(set(old.namelist()) == set(changed.namelist()) and all(old.read(n) == changed.read(n)
                for n in old.namelist() if n != "manifest.json"), "Training rows changed")
    weights = root/"discovery-library"
    ledger = out/"physical.jsonl"
    settings = env(weights, ledger)
    for key in list(settings):
        if key.startswith("SUPERAGENT_"):
            del settings[key]
    run_dir = out/"run"
    lock(out/"launch.json", dict(protocol=sha(PLAN), runtime=source_identity(ROOT), original=sha(packet), revised=sha(new_packet),
         unchanged_rows=True, features=sha(query), fit_cap=16, seconds=1800, model_hint=None, policy_hint=None))
    for action, args in (("solve", ["solve",new_packet,"--fit-limit",16,"--time-limit",1800,"--prediction-input",query]),
                         ("reload", ["predict",query])):
        with (out/(action+".log")).open("x") as log:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,args),"--run-dir",str(run_dir),
                            "--weights-dir",str(weights),"--output",str(out/(action+".csv"))],
                           cwd=ROOT,env=settings,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=1980)
    agent = run_dir/"agent"
    outcome, review = read(agent/"run_outcome.json"), read(agent/"analogy_review.json")
    contract = read(agent/"task_contract.json")
    fits = len(ledger.read_text().splitlines())
    require(contract["metric"]["labels"] == [1,0] and contract["metric"]["name"] == "accuracy",
            "Reordered classes did not reach the agent")
    require(outcome["status"] == "completed" and outcome["completed_classifier_fits"] == fits <= 16
            and outcome["attempted_classifier_fits"] == fits and outcome["new_api_requests"] == 0
            and sha(out/"solve.csv") == sha(out/"reload.csv"), "Incomplete delivery")
    selection = read(agent/"final_selection.json")
    lock(out/"seal.json", dict(protocol=sha(out/"launch.json"), metric=contract["metric"], adviser=review["status"],
         nominated_share=review.get("locked_weight"), deployed_share=deployed_share(selection, review.get("candidate")),
         members=outcome["members"], fits=fits, api_requests=0, predictions=sha(out/"solve.csv"),
         independent_reload=True, final_model=selection["model_sha256"]))
    print("reordered movie task",read(out/"seal.json"),flush=True)


def compare(root, out):
    before = root/"movie-adviser/revised"
    ordinary, reordered = read(before/"sealed.json"), read(out/"seal.json")
    def predictions(path, expected):
        require(sha(path) == expected, "Sealed predictions changed")
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    a = predictions(before/"prediction.csv", ordinary["predictions"])
    b = predictions(out/"solve.csv", reordered["predictions"])
    require(len(a) == len(b) and [r["row_id"] for r in a] == [r["row_id"] for r in b], "Row mismatch")
    record = dict(ordinary=ordinary, reordered=reordered, rows=len(a),
                  predictions_changed=sum(x["prediction"]!=y["prediction"] for x,y in zip(a,b)),
                  held_labels_opened=False, course_validation_opened=False, H2_opened=False)
    lock(out/"comparison.json", record)
    print(record["rows"],record["predictions_changed"],flush=True)


def explain(root, out):
    record = {}
    histories = {}
    for key, folder in (("ordinary", root/"movie-adviser/revised"), ("reordered", out)):
        agent = folder/"run/agent"
        selection = read(agent/"final_selection.json")
        review = read(agent/"analogy_review.json")
        partition = read(agent/"primary_split.json")
        histories[key] = read(agent/"history.json")
        record[key] = dict(selection=sha(agent/"final_selection.json"), review=sha(agent/"analogy_review.json"),
            primary_rows=partition["row_sha256"], primary_labels=partition["label_sha256"],
            nominated_share=review.get("locked_weight"), deployed_share=deployed_share(selection, review["candidate"]))
    fields = ("trial", "candidate", "status", "search_score")
    for first, second in zip(histories["ordinary"], histories["reordered"]):
        if any(first.get(k)!=second.get(k) for k in fields):
            record["first_different_trial"] = {name:{k:row.get(k) for k in (*fields, "error")}
                for name,row in (("ordinary",first),("reordered",second))}
            break
    lock(out/"route_evidence.json",dict(comparison=sha(out/"comparison.json"), details=record, held_labels_opened=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","compare","explain"))
    parser.add_argument("--root",type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument("--source",type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    study=args.root/"adviser-class-order"
    if args.action=="run":
        study.mkdir(exist_ok=False)
        replay(args.root,study)
        solve(args.root,args.source,study)
    elif args.action=="compare":
        compare(args.root,study)
    else:
        explain(args.root,study)
