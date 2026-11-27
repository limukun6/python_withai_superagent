"""Confirm the frozen 48-fit ordinary catalogue on the two remaining course-training folds."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_fusion_match import prepare
from scripts.study_task1_outer import compare, prediction_csv
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/ordinary_catalogue_confirmation.json"
ARMS = ("partner", "regular")
NEW_FOLDS = (3, 4)


def grade(previous, out):
    source = read(previous / "prepared.json")["source"]
    previous_result = read(previous / "result.json")
    expected = previous_result["folds"][0]["methods"]
    solutions = {}
    records = []
    seals = {}
    for index in range(5):
        root = previous if index < 3 else out
        fold = root / str(index)
        geometry = read(root / "prepared.json")
        info = read(fold / "inputs.json")
        require(geometry["source"] == source and geometry["folds"][str(index)] == sha(fold / "inputs.json")
                and info["outer_answers"] == sha(fold / "outer_answers.joblib"), "Unbound outer split")
        if index in NEW_FOLDS:
            require(geometry["protocol"] == sha(PLAN), "Unexpected confirmation protocol")
        methods = {}
        for arm in ARMS:
            area = fold / arm
            seal = read(area / "sealed.json")
            launch = read(area / "launch.json")
            require(seal["launch"] == sha(area / "launch.json") and launch["cap"] == 48
                    and launch["source"] == expected[arm]["source"] and launch["feedback_mode"] is None
                    and launch["method_hint"] is None and seal["fits"] <= 48 and seal["independent_reload"]
                    and seal["status"] == "completed" and not seal["model_api_requests"]
                    and seal["prediction"] == sha(area / "solve.csv") == sha(area / "reload.csv"),
                    "Unverified ordinary replay")
            solutions[index, arm] = prediction_csv(area / "solve.csv", info["roles"]["external"]["rows"])
            methods[arm] = dict(members=seal["members"], fits=seal["fits"], seconds=round(seal["seconds"], 2))
            seals[f"{index}/{arm}"] = sha(area / "sealed.json")
        records.append(dict(fold=index, methods=methods))
    save(out / "scoring_lock.json", dict(protocol=sha(PLAN), previous=sha(previous / "result.json"), seals=seals))
    observed = []
    for index, record in enumerate(records):
        fold = (previous if index < 3 else out) / str(index)
        answer = joblib.load(fold / "outer_answers.joblib")
        truth = answer["y"].astype(str)
        old, new = (solutions[index, arm] == truth for arm in ARMS)
        observed.append((new, old, answer["groups"]))
        record.update(rows=len(truth), correct=dict(partner=int(old.sum()), regular=int(new.sum())),
                      paired=compare(new, old, answer["groups"]))
    def summarize(observations):
        new, old, groups = (np.concatenate(items) for items in zip(*observations, strict=True))
        return dict(rows=len(groups), partner=int(old.sum()), regular=int(new.sum()), **compare(new, old, groups))
    fresh = summarize(observed[3:])
    combined = summarize(observed)
    eligible = all(records[i]["paired"]["net"] >= 0 for i in NEW_FOLDS) and min(fresh["gain"], combined["gain"]) >= 0.003
    result = dict(protocol=sha(PLAN), lock=sha(out / "scoring_lock.json"), folds=records,
                  confirmation=fresh, five_folds=combined, advance=eligible, model_api_requests=0,
                  designated_validation_opened=False, official_test_opened=False)
    save(out / "result.json", result)
    print(fresh, combined, eligible, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "grade"))
    parser.add_argument("--base", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    target = args.base / "ordinary_catalogue"
    if args.action == "prepare":
        prepare(args.base / "remaining_folds_source", target, indices=NEW_FOLDS, protocol=PLAN,
                seed_label="AIAA full training extended evidence panels v1")
    else:
        grade(args.base / "remaining_folds", target)
