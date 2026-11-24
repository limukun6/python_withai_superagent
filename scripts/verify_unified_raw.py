"""Verify two cold runs of the raw-package entry with the original validation rows."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from scripts.unified_released_study import (
    EVIDENCE, ROOT, archive, env, make_zip, predictions, read, require, save, sha, title_rows,
)
from superagent.ab_run import sources
from superagent.prediction_evidence import fingerprint

TAGS = ("table", "titles")
MEMBERS = {
    "table": ["in_vehicle_coupon_recommendation/" + f for f in ("C_train.npy", "y_train.npy", "Readme.md")],
    "titles": ["Released_Task2-Headlines_CTR/train.csv", "Released_Task2-Headlines_CTR/README.md"],
}
PLAN = ROOT / "docs/batches/unified_raw_protocol_20261124_zh.md"


def signatures():
    return {"entrypoint": sha(ROOT / "unified_agent.py"), "study": sha(Path(__file__)), "plan": sha(PLAN),
            "observer": sha(ROOT / "scripts/unified_acceptance/sitecustomize.py"),
            "inner_sources": hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()}


def execute(args, log, environment):
    with log.open("x") as stream:
        run = subprocess.run([sys.executable, "-m", "unified_agent", *map(str, args)], cwd=ROOT, env=environment,
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=1450)
    require(run.returncode == 0, "Raw entry failed; see " + str(log))


def initiate(a):
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / "protocol.json", {"sources": signatures(), "archives": {k: sha(archive(a.packages, k)) for k in TAGS},
                                   "fit_limit_per_run": 10, "model_api_limit": 0, "designated_test": False})


def train(a, tag):
    protocol = read(a.out / "protocol.json")
    package = archive(a.packages, tag)
    require(protocol["sources"] == signatures() and protocol["archives"][tag] == sha(package), "Frozen inputs changed")
    folder = a.out / tag
    folder.mkdir()
    ep = env(a.weights, folder / "physical.jsonl")
    ep.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / "scripts/unified_acceptance", ROOT))),
              UNIFIED_RAW_INPUT=str(package), UNIFIED_RAW_LOG=str(folder / "raw_members.jsonl"),
              UNIFIED_RAW_MEMBERS=json.dumps(MEMBERS[tag]))
    execute(["train", package, "--run-dir", folder / "run"], folder / "train.log", ep)
    native = folder / "run/agent"
    sel, outcome = read(native / "final_selection.json"), read(native / "run_outcome.json")
    training = read(folder / "run/training_protocol.json")
    workers = [json.loads(s) for s in (folder / "physical.jsonl").read_text().splitlines()]
    seen = [json.loads(s) for s in (folder / "raw_members.jsonl").read_text().splitlines()]
    counts = read(native / "classifier_fits.json")
    require(set(seen) == set(MEMBERS[tag]) and
            training["entrypoint_sha256"] == protocol["sources"]["entrypoint"] and
            training["model_training_rows"] == sel["training_rows"] and
            sel["model_sha256"] == sha(native / "selected_model.joblib") and
            outcome["source_unchanged"] and outcome["requested_planner_completed"] and outcome["full_refit_completed"] and
            not outcome["formal_test_accessed"] and outcome["new_api_requests"] == 0 and
            len(workers) == counts["completed_classifier_fits"] == counts["attempted_classifier_fits"] <= 10,
            "Raw access, selection or fits did not reconcile")
    audit = read(folder / "run/audit.json") if training["audit_rows"] else None
    if audit:
        require(audit["model_sha256"] == sel["model_sha256"] and audit["rows"] == training["audit_rows"] and
                training["audit_row_ids_sha256"] == fingerprint(read(EVIDENCE / "title_reserve_ids.json")),
                "Automatic audit does not reproduce the historical grouping")
    save(folder / "completion.json", {"training": training, "model": sel["model_sha256"], "winner": sel["family"],
                                      "members": sel["ensemble_components"], "fits": workers, "api_calls": 0,
                                      "source_members_opened": seen, "actions": read(native / "actions.json"),
                                      "independent_training_audit": audit})
    print(tag, sel["family"], len(workers), flush=True)


def feature_package(packages, tag, output):
    if tag == "table":
        with zipfile.ZipFile(archive(packages, tag)) as course:
            make_zip(output, [("C_test.npy", course.read("in_vehicle_coupon_recommendation/C_val.npy"))])
        return
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["headline_A", "headline_B"])
    writer.writerows(title_rows(archive(packages, tag), "val", labels=False))
    make_zip(output, [("test.csv", buf.getvalue().encode())])


def score(a):
    require(read(a.out / "protocol.json")["sources"] == signatures(), "Frozen source differs")
    checked = {k: read(a.out / k / "completion.json") for k in TAGS}
    for k in TAGS:
        require(checked[k]["model"] == sha(a.out / k / "run/agent/selected_model.joblib"), "Unsealed model")
    stage = a.out / "designated"
    stage.mkdir()
    sealed = {}
    for tag in TAGS:
        feat = stage / (tag + ".zip")
        feature_package(a.packages, tag, feat)
        digests = []
        for i in range(2):
            dest = stage / f"{tag}_{i}" / "predictions.csv"
            execute(["predict", feat, "--run-dir", a.out / tag / "run", "--output", dest],
                    stage / f"{tag}_{i}.log", env(a.weights, stage / "unexpected_fit.jsonl"))
            digests.append(sha(dest))
        require(digests[0] == digests[1] and not (stage / "unexpected_fit.jsonl").exists(), "Prediction replay changed")
        sealed[tag] = digests
    save(stage / "predictions_frozen.json", sealed)
    results = {}
    for tag in TAGS:
        if tag == "table":
            with zipfile.ZipFile(archive(a.packages, tag)) as z:
                truth = np.load(io.BytesIO(z.read("in_vehicle_coupon_recommendation/y_val.npy")), allow_pickle=False)
        else:
            truth = np.asarray([r[2] for r in title_rows(archive(a.packages, tag), "val", labels=True)])
        hard = predictions(stage / f"{tag}_0/predictions.csv", len(truth))
        historical = predictions(EVIDENCE / f"validation/reference_{tag}_0/predictions.csv", len(truth))
        correct = int((hard == truth).sum())
        results[tag] = {"correct": correct, "rows": len(truth), "accuracy": correct / len(truth),
                        "identical_to_historical_rows": int((hard == historical).sum()), "prediction": sealed[tag][0],
                        "model": checked[tag]["model"], "source": "course designated validation"}
    save(a.out / "results.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", *TAGS, "score"))
    for name in ("out", "packages", "weights"):
        parser.add_argument("--" + name, required=True, type=lambda x: Path(x).resolve())
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        initiate(arguments)
    elif arguments.stage == "score":
        score(arguments)
    else:
        train(arguments, arguments.stage)
