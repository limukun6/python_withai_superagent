"""Prospectively assess two 64-fit ordinary agents on blinded partitions of course training data."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_fusion_match import prepare
from scripts.study_task1_outer import compare, prediction_csv
from scripts.study_task1_remaining_folds import signature
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/seed_policy_qualification.json"
FOLDS = (0, 2, 4)
ARMS = ("regular", "replicated")


def grade(out):
    geometry = read(out / "prepared.json")
    require(geometry["protocol"] == sha(PLAN), "Wrong confirmation protocol")
    seals, records, predictions = {}, [], {}
    for idx in FOLDS:
        folder = out / str(idx)
        info = read(folder / "inputs.json")
        require(sha(folder / "inputs.json") == geometry["folds"][str(idx)] and sha(folder / "outer_answers.joblib") == info["outer_answers"],
                "Unverified outer packets")
        methods = {}
        for arm in ARMS:
            side = folder / arm
            launch = read(side / "launch.json")
            receipt = read(side / "sealed.json")
            require(launch["cap"] == 64 and launch["feedback_mode"] == ("seed-replication" if arm == "replicated" else None)
                    and launch["inputs"] == sha(folder / "inputs.json") and launch["method_hint"] is None
                    and receipt["launch"] == sha(side / "launch.json") and receipt["independent_reload"]
                    and receipt["status"] == "completed" and receipt["fits"] <= 64 and not receipt["model_api_requests"]
                    and sha(side / "solve.csv") == receipt["prediction"] == sha(side / "reload.csv"), "Unverified delivery")
            seals[f"{idx}/{arm}"] = sha(side / "sealed.json")
            predictions[idx, arm] = prediction_csv(side / "solve.csv", info["roles"]["external"]["rows"])
            methods[arm] = dict(source=launch["source"], members=receipt["members"], fits=receipt["fits"], seconds=receipt["seconds"])
        require(methods["regular"]["source"] == methods["replicated"]["source"], "Different executable source")
        same = signature(folder / "regular") == signature(folder / "replicated")
        require(same, "Primary and confirmation histories diverged before the replicate policy")
        review = read(folder / "replicated/run/agent/seed_replication.json")
        select = read(folder / "replicated/run/agent/final_selection.json")
        require(review["action"] in ("adopt", "retain") and (review["action"] != "adopt" or review["model"] == select["model_sha256"]),
                "Unverified replication choice")
        records.append(dict(fold=idx, methods=methods, same_initial_history=same, review=review))
    save(out / "scoring_lock.json", dict(protocol=sha(PLAN), geometry=sha(out / "prepared.json"), solves=seals))
    left, right, groups = [], [], []
    for item in records:
        idx = item["fold"]
        answer = joblib.load(out / str(idx) / "outer_answers.joblib")
        y = answer["y"].astype(str)
        old, new = (predictions[idx, k] == y for k in ARMS)
        item.update(rows=len(y), regular=int(old.sum()), replicated=int(new.sum()), paired=compare(new, old, answer["groups"]))
        left.extend(old)
        right.extend(new)
        groups.extend(answer["groups"])
    pooled = compare(np.asarray(right), np.asarray(left), groups)
    gains = [r["paired"]["gain"] for r in records]
    advance = pooled["gain"] >= .003 and sum(g > 0 for g in gains) >= 2 and min(gains) >= -.003
    result = dict(lock=sha(out / "scoring_lock.json"), folds=records, rows=len(groups),
                  correct=dict(regular=int(np.sum(left)), replicated=int(np.sum(right))),
                  pooled=pooled, advance=bool(advance), api_requests=0, designated_validation_opened=False, official_test_opened=False)
    save(out / "result.json", result)
    print(result["correct"], pooled, advance, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "grade"))
    parser.add_argument("--base", required=True, type=lambda p:Path(p).resolve())
    a = parser.parse_args()
    out = a.base / "seed_policy"
    if a.action == "prepare":
        prepare(a.base / "remaining_folds_source", out, indices=FOLDS, protocol=PLAN,
                seed_label="AIAA released training fresh seed policy qualification v1")
    else:
        grade(out)
