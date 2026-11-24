"""Exercise the course train/predict CLI with the same unnamed package in two source checkouts."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score

from scripts.unified_released_study import make_zip, read, require, save, sha


def csv_bytes(columns, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    x, y = make_classification(n_samples=620, n_features=8, n_informative=4, n_redundant=2,
                               weights=[.86, .14], random_state=1124)
    columns = [f"field_{i}" for i in range(x.shape[1])]
    make_zip(folder/"train.zip", [
        ("README.md", b"Primary metric: macro-F1.\n"), ("train.csv", csv_bytes([*columns, "label"],
         ([*row, int(label)] for row, label in zip(x[:460], y[:460], strict=True))))])
    make_zip(folder/"features.zip", [("test.csv", csv_bytes(columns, x[460:]))])
    save(folder/"audit_truth.json", y[460:].tolist())


def run(folder, source, out):
    out.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, AB_B1_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       SUPERAGENT_OFFLINE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       PYTHONPATH=os.pathsep.join(map(str, (source/"scripts/ab_acceptance_guard", source))))
    root = out/"run"
    def execute(log, *args):
        with (out/log).open("x") as stream:
            subprocess.run([sys.executable, "-m", "unified_agent", *map(str, args)], cwd=source,
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    execute("train.log", "train", folder/"train.zip", "--run-dir", root)
    source_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    agent = root/"agent"
    outcome = read(agent/"run_outcome.json")
    require(outcome["full_refit_completed"] and outcome["new_api_requests"] == 0, "Incomplete local path")
    completed = [r for r in read(agent/"history.json") if r["status"] == "completed"]
    metric = read(agent/"task_spec.json")["metric_spec"]["name"]
    require(completed and all(r["primary_metric"] == metric for r in completed)
            and outcome["primary_metric"] == metric, "Inconsistent trial or final metric")
    outputs = []
    for i in range(2):
        output = out/f"prediction_{i}"/"predictions.csv"
        execute(f"predict_{i}.log", "predict", folder/"features.zip", "--run-dir", root, "--output", output)
        outputs.append(output)
    require(sha(outputs[0]) == sha(outputs[1]), "Fresh-process predictions differ")
    frozen = dict(source=source_id, train_package=sha(folder/"train.zip"), prediction=sha(outputs[0]),
                  features=sha(folder/"features.zip"), model=sha(agent/"selected_model.joblib"))
    save(out/"frozen.json", frozen)
    with outputs[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    y = read(folder/"audit_truth.json")
    require([int(r["row_id"]) for r in rows] == list(range(len(y))), "Mismatched row order")
    prediction = [int(r["prediction"]) for r in rows]
    counter = read(agent/"classifier_fits.json")
    require(counter["attempted_classifier_fits"] == counter["completed_classifier_fits"] <= 10
            and not counter["unknown_failed_attempts"], "Classifier budget exceeded")
    result = dict(frozen=frozen, train_rows=460, audit_rows=len(y), requested_metric="macro_f1",
                  actual_metric=metric, source=read(agent/"task_spec.json")["goal_sources"]["primary_metric"],
                  trial_metrics=sorted({r["primary_metric"] for r in completed}), selected=outcome["selected_candidate"],
                  fits=counter["attempted_classifier_fits"], api=outcome["new_api_requests"],
                  audit_accuracy=float(accuracy_score(y, prediction)), audit_macro_f1=float(f1_score(y, prediction, average="macro")),
                  audit_correct=int(np.sum(np.asarray(y) == prediction)), matched_exports=len(outputs))
    save(out/"result.json", result)
    print(metric, result["selected"], result["fits"], result["audit_macro_f1"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("folder", type=Path)
    compare = sub.add_parser("run")
    for arg in ("folder", "source", "out"):
        compare.add_argument(arg, type=lambda v: Path(v).resolve())
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.folder)
    else:
        run(args.folder, args.source, args.out)
