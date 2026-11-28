"""Ask whether unseen-profile feedback supports vetoing the ordinary repeated-seed vote."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_accuracy_veto import accuracy_pass
from scripts.study_task1_outer import compare, prediction_csv
from scripts.study_task1_population_expert import STAGE as POPULATION, PHASES
from scripts.study_task1_unseen_profiles import FOLDS, STUDY
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/novel_seed_feedback.json"
STAGE = "novel_seed_feedback_1128"


def elect(out):
    root = out / STAGE
    root.mkdir(exist_ok=False)
    locks = {}
    prior = read(out / STUDY / "score_lock.json")
    for i in FOLDS:
        area = root / str(i)
        area.mkdir()
        bank, fold = out / POPULATION / str(i), out / STUDY / str(i)
        rows = [read(bank / f"{p}-seed-review.json") for p in PHASES]
        require(all(r["protocol"] == sha(PLAN) for r in rows), "Seed review belongs to another plan")
        panels = [r["before_pilot"] for r in rows]
        rc = read(fold / "replicated/sealed.json")
        choice = read(fold / "replicated/run/agent/seed_replication.json")
        eligible = choice["action"] == "adopt"
        use = (eligible and accuracy_pass(panels) and sum(v["rows"]*v["brier_change"] for v in panels) <= 0)
        features = bank / "sealed_features.npz"
        require(sha(features) == read(bank / "built.json")["probabilities"], "Novelty mask changed")
        with np.load(features, allow_pickle=False) as features_file:
            novel = features_file["novel"]
        default = fold / "before_pilot.npy"
        require(sha(default) == prior["solves"][str(i)]["original"] and sha(fold / "replicated/solve.csv") == rc["prediction"],
                "Mismatched parent models")
        orig = np.load(default, allow_pickle=False)
        normal = prediction_csv(fold / "replicated/solve.csv", len(novel))
        dst = area / "chosen.npy"
        np.save(dst, np.where(novel & use, orig, normal), allow_pickle=False)
        decision = area / "decision.json"
        save(decision, dict(protocol=sha(PLAN), eligible=eligible, restore=bool(use), prediction=sha(dst),
                            panel_nets=[v["net"] for v in panels], novel_rows=int(novel.sum()), parent=rc["prediction"],
                            panel_feedback={p:sha(bank / f"{p}-seed-review.json") for p in PHASES}))
        locks[str(i)] = sha(decision)
    save(root / "score_lock.json",dict(protocol=sha(PLAN),folds=locks))


def grade(out):
    root = out / STAGE
    lock = read(root / "score_lock.json")
    rows, groups, reference, chosen = [], [], [], []
    for i in FOLDS:
        area, source = root / str(i), out / STUDY / str(i)
        rec = read(area / "decision.json")
        predicted, baseline = area / "chosen.npy", source / "replicated/solve.csv"
        answer = source / "outer_answers.joblib"
        require(sha(area / "decision.json") == lock["folds"][str(i)] and sha(predicted) == rec["prediction"]
                and sha(baseline) == rec["parent"] and sha(answer) == read(source / "inputs.json")["external_answers"],
                "Sealed predictions or answer packet differ")
        packet = joblib.load(answer)
        y = packet["y"].astype(str)
        p = np.load(predicted, allow_pickle=False)
        require(p.shape == y.shape, "Unexpected output rows")
        a, b = prediction_csv(baseline,len(y)) == y, p == y
        groups.extend(packet["groups"])
        chosen.extend(b)
        reference.extend(a)
        rows.append(dict(fold=i, rows=len(y), adopted_repeats=rec["eligible"], restore=rec["restore"], panel_nets=rec["panel_nets"],
                         ordinary=int(a.sum()), proposed=int(b.sum())))
    paired = compare(np.asarray(chosen),np.asarray(reference),groups)
    advance = (sum(r["adopted_repeats"] for r in rows)>1 and any(r["proposed"]>r["ordinary"] for r in rows)
               and all(r["proposed"]>=r["ordinary"] for r in rows) and paired["gain"]>.003)
    result = dict(lock=sha(root / "score_lock.json"), folds=rows, rows=len(groups),
                  correct=dict(ordinary=int(np.sum(reference)), chosen=int(np.sum(chosen))), paired=paired,
                  advance=advance, new_fits=0, model_api_requests=0, designated_validation_opened=False,
                  official_test_opened=False, h2_opened=False)
    save(root / "result.json", result)
    print("seed feedback", result["correct"], rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("elect","grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    {"elect":elect, "grade":grade}[args.action](args.out)
