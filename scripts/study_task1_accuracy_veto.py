"""Replay the saved decisions with the secondary Brier veto switched off."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_conditional_opportunity import STUDY, load_matrix
from scripts.study_task1_disagreement_reading import HEADS, STAGE as HEAD_STAGE
from scripts.study_task1_outer import arrays, compare
from scripts.study_task1_seed_completion import origin
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/accuracy_veto_ablation.json"
STAGE = "accuracy_veto_ablation_1128"


def accuracy_pass(panels):
    return min(r["net"] for r in panels) >= 0 and sum(r["net"] for r in panels) >= .002 * sum(r["rows"] for r in panels)


def replay(out):
    root = out / STAGE
    root.mkdir(exist_ok=False)
    locks = {}
    for i in range(5):
        area = root / str(i)
        area.mkdir()
        src = out / STUDY / str(i)
        heads = out / HEAD_STAGE / str(i)
        cond, review = (read(d / "decision.json") for d in (src, heads))
        for d in (src, heads):
            require(read(d / "sealed.json")["decision"] == sha(d / "decision.json"), "Decision evidence changed")
        c = [cond["nomination"], cond["confirmation"]]
        eligible = cond["weight"] > 0 and accuracy_pass(c)
        brier_good = sum(v["rows"] * v["brier_change"] for v in c) <= 0
        require((eligible and brier_good) == (cond["action"] == "adopt"), "Recorded candidate decision differs")
        base = load_matrix(src / "control", "external")
        w = cond["weight"] if eligible else 0.
        p = (1-w)*base + w*load_matrix(src / cond["nominee"], "external")
        train = origin(out, i) / "anonymous/training.zip"
        require(sha(train) == read(origin(out, i) / "inputs.json")["entry_inputs"], "Training input changed")
        labels, = arrays(train, ["y_train.npy"])
        classes = np.unique(labels).astype(str)
        ranking = review["cross_checks"]
        passed = [h for h in HEADS if accuracy_pass(ranking[h]["folds"])]
        rank = lambda h: (-ranking[h]["net"], ranking[h]["brier_change"], HEADS.index(h))
        strict = min((h for h in passed if ranking[h]["brier_change"] <= 0), key=rank, default="retain")
        loose = min(passed, key=rank, default="retain")
        require(strict == review["chosen"], "Recorded head decision differs")
        head_prediction = heads / (loose + ".npy")
        require(sha(head_prediction) == read(heads / "sealed.json")["predictions"][loose], "Head prediction changed")
        base_labels = classes[(base > .5).astype(int)]
        require(np.array_equal(base_labels, np.load(heads / "retain.npy", allow_pickle=False)), "Controls differ")
        sealed = {}
        for n, pred in dict(control=base_labels, candidate=classes[(p > .5).astype(int)],
                            head=np.load(head_prediction, allow_pickle=False)).items():
            file = area / (n + ".npy")
            np.save(file, pred, allow_pickle=False)
            sealed[n] = sha(file)
        save(area / "decision.json", dict(candidate=cond["nominee"], accepted_weight=w, strict=cond["action"],
             head=loose, strict_head=strict, candidate_net=[v["net"] for v in c], predictions=sealed,
             input_decisions=[sha(d / "decision.json") for d in (src, heads)]))
        locks[str(i)] = sha(area / "decision.json")
    save(root / "score_lock.json", dict(protocol=sha(PLAN), folds=locks))


def grade(out):
    root = out / STAGE
    lock = read(root / "score_lock.json")
    groups, rows, hits = [], [], {}
    for i in range(5):
        area = root / str(i)
        fold = origin(out, i)
        file = area / "decision.json"
        require(sha(file) == lock["folds"][str(i)], "Unsealed decision")
        decision = read(file)
        answers = fold / "outer_answers.joblib"
        require(sha(answers) == read(fold / "inputs.json")["outer_answers"], "Changed answers")
        packet = joblib.load(answers)
        y = packet["y"].astype(str)
        groups.extend(packet["groups"])
        correct = {}
        for n, digest in decision["predictions"].items():
            p = area / (n + ".npy")
            require(sha(p) == digest, "Sealed predictions changed")
            labels = np.load(p, allow_pickle=False)
            require(labels.shape == y.shape, "Wrong prediction length")
            bit = labels == y
            hits.setdefault(n, []).extend(bit)
            correct[n] = int(bit.sum())
        rows.append(dict(fold=i, rows=len(y), correct=correct, candidate=decision["candidate"],
                         accepted_weight=decision["accepted_weight"], head=decision["head"]))
    compared = {k: compare(np.asarray(vals), np.asarray(hits["control"]), groups) for k, vals in hits.items() if k != "control"}
    result = dict(lock=sha(root / "score_lock.json"), folds=rows, rows=len(groups),
                  correct={k:int(np.sum(v)) for k,v in hits.items()}, compared=compared,
                  new_fits=0, model_api_requests=0, designated_validation_opened=False, official_test_opened=False, h2_opened=False)
    save(root / "result.json", result)
    print(result["correct"], compared, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    {"replay": replay, "grade": grade}[args.action](args.out)
