"""Finish five-fold coverage using the pre-pilot full-data export as a within-run control."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_fusion_match import prepare
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_policy import FOLDS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.models import predict_with_model

PLAN = ROOT / "research/task1_feedback_extension_1127/seed_policy_completion.json"
EXTRA = (1, 3)


def origin(root, i):
    return root / ("seed_policy" if i in FOLDS else "seed_completion") / str(i)


def seal_originals(out):
    locks = {}
    for i in range(5):
        folder = origin(out, i)
        info, side = read(folder / "inputs.json"), folder / "replicated"
        receipt, launch = read(side / "sealed.json"), read(side / "launch.json")
        require(launch["cap"] == 64 and launch["feedback_mode"] == "seed-replication" and launch["method_hint"] is None
                and launch["inputs"] == sha(folder / "inputs.json") and sha(side / "launch.json") == receipt["launch"]
                and receipt["prediction"] == sha(side / "solve.csv") == sha(side / "reload.csv")
                and receipt["independent_reload"] and receipt["status"] == "completed" and not receipt["model_api_requests"],
                "Invalid ordinary entry or replay")
        q = folder / "anonymous/queries.zip"
        require(sha(q) == info["entry_queries"], "Wrong inference packet")
        rows, = arrays(q, ["C_test.npy"])
        src = side / "run/agent"
        report = read(src / "seed_replication.json")
        model_file = src / "selected_model.joblib"
        require(sha(model_file) == report["incumbent"], "Original export no longer matches the pre-pilot identity")
        bundle = joblib.load(model_file)
        labels = np.asarray(predict_with_model(bundle["kind"], bundle["model"], rows)).astype(str)
        require(len(labels) == len(rows) == receipt["rows"], "Incomplete original inference")
        if i in FOLDS:
            regular = folder / "regular"
            require(read(regular / "sealed.json")["prediction"] == sha(regular / "solve.csv")
                    and np.array_equal(prediction_csv(regular / "solve.csv", len(rows)), labels),
                    "Within-run control differs from separate ordinary entry")
        orig = out / "seed_completion" / ("original_" + str(i) + ".npy")
        np.save(orig, labels, allow_pickle=False)
        locks[str(i)] = dict(original=sha(orig), proposed=receipt["prediction"], receipt=sha(side / "sealed.json"),
                             model=report["incumbent"], source=launch["source"], action=report["action"],
                             fits=receipt["fits"], members=receipt["members"], external=info["outer_answers"])
    require(len({r["source"] for r in locks.values()}) == 1, "Five runs did not use the same source")
    save(out / "seed_completion/score_lock.json", dict(protocol=sha(PLAN), initial_result=sha(out / "seed_policy/result.json"), folds=locks))


def grade(out):
    root = out / "seed_completion"
    data = read(root / "score_lock.json")
    records, groups = [], []
    controls, proposed, seen = [], [], set()
    for i in range(5):
        folder, seal = origin(out, i), data["folds"][str(i)]
        answers = folder / "outer_answers.joblib"
        original, treatment = root / ("original_" + str(i) + ".npy"), folder / "replicated/solve.csv"
        require(sha(answers) == seal["external"] and sha(original) == seal["original"] and sha(treatment) == seal["proposed"],
                "Modified assessment packet")
        packet = joblib.load(answers)
        truth = packet["y"].astype(str)
        distinct = set(packet["groups"])
        require(not seen.intersection(distinct), "Overlapping complete feature groups between outer folds")
        seen.update(distinct)
        groups.extend(packet["groups"])
        a = np.load(original, allow_pickle=False) == truth
        b = prediction_csv(treatment, len(truth)) == truth
        controls.extend(a)
        proposed.extend(b)
        records.append(dict(fold=i, rows=len(a), regular=int(a.sum()), replicated=int(b.sum()),
             control="separate cold solve and matching exported incumbent" if i in FOLDS else "exported incumbent",
             action=seal["action"], members=seal["members"], fits=seal["fits"], paired=compare(b,a,packet["groups"])))
    require(len(groups) == 8117, "Five-fold coverage is incomplete")
    result = dict(lock=sha(root / "score_lock.json"), folds=records, rows=len(groups), correct=dict(regular=int(np.sum(controls)),
         replicated=int(np.sum(proposed))), pooled=compare(np.asarray(proposed), np.asarray(controls), groups),
         further_cold_fits=sum(r["fits"] for r in records if r["fold"] in EXTRA), designated_gate=read(out / "seed_policy/result.json")["advance"],
         api_requests=0, official_test_opened=False)
    save(root / "result.json", result)
    print(result["correct"], result["pooled"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out / "remaining_folds_source", args.out / "seed_completion", indices=EXTRA, protocol=PLAN,
                seed_label="AIAA released training fresh seed policy qualification v1")
    elif args.action == "seal":
        seal_originals(args.out)
    else:
        grade(args.out)
