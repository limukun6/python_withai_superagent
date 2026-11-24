"""Compare full and bounded label export from the same models without opening any outer labels."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from scripts.bounded_preference_delivery import package
from scripts.run_conditional_handoff import load_export
from scripts.study_brier_replication import RULE
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_brier_replication import PUBLIC as PARENT
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from superagent.ab_complement import canonical_probabilities
from superagent.bounded_vote import BoundedPreferenceVote
from unified_agent import main as unified_main

PLAN = ROOT / "docs/batches/preference_bounded_vote_plan_20261124_zh.md"
PUBLIC = ROOT / "research/bounded_preference_vote_1124"


def profile(kind, features, run, weights, out):
    calls, stats = [], []
    def observe(model, rows):
        calls.append(len(rows))
        return canonical_probabilities(model, rows)
    actual = BoundedPreferenceVote.predict_with_counts
    def bounded(self, rows):
        labels, metrics = actual(self, rows)
        stats.append(metrics)
        return labels, metrics
    output = out / "predictions.csv"
    args = ["unified_agent", "predict", str(features), "--run-dir", str(run), "--weights", str(weights), "--output", str(output)]
    with patch("superagent.ab_complement.canonical_probabilities", observe), patch("superagent.bounded_vote.canonical_probabilities", observe), \
            patch.object(BoundedPreferenceVote, "predict_with_counts", bounded), patch.object(sys, "argv", args):
        require(unified_main() == 0, "Native exporter did not finish")
    n = len(load_export(output)["row_id"])
    require(calls[:2] == [n, n], "Reference members were not evaluated over the same full input")
    if kind == "full":
        require(calls == [n, n, n] and not stats, "Unexpected full-vote execution")
        record = dict(rows=n, certified=0, third_rows=n, full_batch_rechecks=0)
    else:
        require(len(stats) == 1 and stats[0]["third_rows"] == sum(calls[2:])
                and stats[0]["full_batch_rechecks"] == int(len(calls) == 4), "Bounded call count differs")
        record = stats[0]
    save(out / "work.json", {**record, "component_batch_rows": calls, "export": sha(output)})


def run_cohort(out, local, weights):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AB_B1_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SUPERAGENT_OFFLINE="1",
               PYTHONPATH=os.pathsep.join(map(str,(ROOT / "scripts/ab_acceptance_guard", ROOT))))
    env.pop("SUPERAGENT_PORTFOLIO_CACHE", None)
    records = {}
    source_protocol = read(PARENT / "protocol.json")
    for i in range(5):
        key = f"fold_{i}"
        src = local / key
        public = PARENT / key
        sel = read(public / RULE / "run/agent/final_selection.json")
        if len(sel["ensemble_components"]) != 3:
            continue
        original = src / RULE / "run"
        features = src / "features.zip"
        require(sha(features) == source_protocol["folds"][str(i)]["features"] and
                sha(original / "agent/final_selection.json") == sha(public / RULE / "run/agent/final_selection.json")
                and sha(original / "agent/selected_model.joblib") == sel["model_sha256"], "Mismatched original input or bundle")
        fold = out / key
        fold.mkdir(exist_ok=True)
        optimized = fold / "optimized"
        if not optimized.exists():
            package(original, optimized)
        opt_selection = read(optimized / "agent/final_selection.json")
        require(opt_selection["original_model_sha256"] == sel["model_sha256"] and
                opt_selection["task_spec_sha256"] == sel["task_spec_sha256"] and
                opt_selection["model_sha256"] == sha(optimized / "agent/selected_model.joblib"), "Different optimized delivery")
        order = ["bounded", "full"] if len(records) % 2 else ["full", "bounded"]
        for arm in order:
            dest = fold / arm
            if not (dest / "wall.json").exists():
                dest.mkdir(exist_ok=False)
                command = [sys.executable, "-m", "scripts.study_bounded_vote", "profile", arm, str(features),
                           str(original if arm == "full" else optimized), str(weights), str(dest)]
                with (dest / "native.log").open("x") as log:
                    started = time.monotonic()
                    subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    elapsed = time.monotonic() - started
                save(dest / "wall.json", dict(seconds=round(elapsed, 4)))
        dig = sha(public / RULE / "direct/predictions.csv")
        expected = load_export(public / RULE / "direct/predictions.csv")
        arm_records = {}
        for arm in order:
            dest = fold / arm
            values = load_export(dest / "predictions.csv")
            require(np.array_equal(values["row_id"], expected["row_id"]), "Incorrect native output row order")
            rec = read(dest / "work.json")
            require(rec["export"] == sha(dest / "predictions.csv"), "Native output changed")
            arm_records[arm] = dict(work=rec, seconds=read(dest / "wall.json")["seconds"],
                                    label_differences=int(np.sum(values["prediction"] != expected["prediction"])))
        records[key] = dict(order=order, feature_sha256=sha(features), full_reference_export=dig,
                            optimized=opt_selection["model_sha256"], original=sel["model_sha256"], arms=arm_records)
        print(key, [(k, v["work"]["third_rows"], v["seconds"], v["label_differences"]) for k, v in arm_records.items()], flush=True)
    require(len(records) == 4, "Different number of approved three-model deliveries")
    full = sum(r["arms"]["full"]["seconds"] for r in records.values())
    fast = sum(r["arms"]["bounded"]["seconds"] for r in records.values())
    rows = sum(r["arms"]["bounded"]["work"]["rows"] for r in records.values())
    remaining = sum(r["arms"]["bounded"]["work"]["third_rows"] for r in records.values())
    differences = sum(v["label_differences"] for r in records.values() for v in r["arms"].values())
    save(out / "result.json", dict(plan=sha(PLAN), source={f: sha(ROOT / f) for f in
         ("superagent/bounded_vote.py", "scripts/bounded_preference_delivery.py", "scripts/study_bounded_vote.py")},
         folds=records, rows=rows, full_seconds=full, bounded_seconds=fast, third_rows_before=rows, third_rows_after=remaining,
         label_differences=differences, passes=(fast < full and differences == 0), actual_fits=0, model_api_calls=0))


def publish(out):
    copy(out / "result.json", PUBLIC / "result.json")
    for key in read(out / "result.json")["folds"]:
        for part in ("full", "bounded"):
            for name in ("predictions.csv", "work.json", "wall.json"):
                copy(out / key / part / name, PUBLIC / key / part / name)
        f = "optimized/agent/final_selection.json"
        copy(out / key / f, PUBLIC / key / f)
    seal_directory(PUBLIC)


def verify():
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text()
            and checksums(PARENT) == (PARENT / "SHA256SUMS").read_text(), "Changed timing or predictions")
    result = read(PUBLIC / "result.json")
    require(result["plan"] == sha(PLAN) and all(sha(ROOT / f) == v for f, v in result["source"].items()),
            "Different bounded-vote code")
    times = {"full": 0., "bounded": 0.}
    work = {"full": 0, "bounded": 0}
    delta = 0
    for ordinal, (key, trial) in enumerate(result["folds"].items()):
        parent = PARENT / key / RULE
        digest = sha(parent / "direct/predictions.csv")
        expected = load_export(parent / "direct/predictions.csv")
        require(digest == trial["full_reference_export"] and trial["order"] ==
                (["bounded", "full"] if ordinal % 2 else ["full", "bounded"]), "Wrong reference or presentation order")
        normal = read(parent / "run/agent/final_selection.json")
        faster = read(PUBLIC / key / "optimized/agent/final_selection.json")
        require(trial["original"] == normal["model_sha256"] == faster["original_model_sha256"]
                and trial["optimized"] == faster["model_sha256"] and normal["ensemble_components"] == faster["ensemble_components"],
                "Optimized vote differs")
        for arm, record in trial["arms"].items():
            dest = PUBLIC / key / arm
            w = read(dest / "work.json")
            seconds = read(dest / "wall.json")["seconds"]
            observed = load_export(dest / "predictions.csv")
            require(np.array_equal(observed["row_id"], expected["row_id"]), "Row ordering changed")
            errors = int(np.sum(observed["prediction"] != expected["prediction"]))
            require(record["work"] == w and w["export"] == sha(dest / "predictions.csv") and seconds == record["seconds"]
                    and w["rows"] == len(observed["row_id"]) and w["component_batch_rows"][:2] == [w["rows"]] * 2
                    and w["third_rows"] == sum(w["component_batch_rows"][2:]) and errors == record["label_differences"],
                    "Physical workload or label count differs")
            if arm == "bounded":
                require(w["third_rows"] == w["rows"] - w["certified"] + w["full_batch_rechecks"] * w["rows"], "Wrong work bound")
            times[arm] += seconds
            work[arm] += w["third_rows"]
            delta += errors
    require((times["full"], times["bounded"], work["full"], work["bounded"], delta) ==
            (result["full_seconds"], result["bounded_seconds"], result["third_rows_before"], result["third_rows_after"],
             result["label_differences"]) and result["passes"] == (times["bounded"] < times["full"] and not delta),
            "Aggregate timing or outcome differs")
    print("Verified paired execution order, component work, timings and every delivered label.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    run = stages.add_parser("run")
    for arg in ("out", "local", "weights"):
        run.add_argument(arg, type=lambda s: Path(s).resolve())
    one = stages.add_parser("profile")
    one.add_argument("kind", choices=("full", "bounded"))
    for arg in ("features", "run_folder", "weights", "out"):
        one.add_argument(arg, type=Path)
    stages.add_parser("publish").add_argument("out", type=Path)
    stages.add_parser("verify")
    args = parser.parse_args()
    if args.stage == "run":
        run_cohort(args.out, args.local, args.weights)
    elif args.stage == "profile":
        profile(args.kind, args.features, args.run_folder, args.weights, args.out)
    elif args.stage == "publish":
        publish(args.out)
    else:
        verify()
