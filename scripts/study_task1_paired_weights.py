"""Isolate the contribution of complementary training-example weights at the same seed count."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_replicas import SEEDS, probabilities, reseed
from scripts.study_task1_threshold import get_fold
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.splits import sample_groups

PLAN = ROOT / "research/task1_feedback_extension_1127/paired_training_weights.json"


def weights_for(rows):
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=rows))
    _, inverse = np.unique(groups, return_inverse=True)
    raw = np.random.default_rng(700883).uniform(-1., 1., int(inverse.max()) + 1)[inverse]
    centered = raw - raw.mean()
    size = np.max(np.abs(centered))
    if size:
        centered *= .7 / size
    a, b = 1. + centered, 1. - centered
    require(min(a.min(), b.min()) > 0. and np.allclose(a+b, 2.) and np.isclose(a.sum(),len(rows))
            and np.isclose(b.sum(),len(rows)), "Unexpected row influence")
    return a, b


def train(out, i):
    root = out / "paired_weights" / str(i)
    root.mkdir(parents=True, exist_ok=False)
    fold, replica = get_fold(out, 48, i), out / "seed_replication" / str(i)
    src = fold / "regular/run/agent"
    control = read(fold / "regular/sealed.json")
    seed_control = read(replica / "sealed.json")
    require(control["prediction"] == sha(fold / "regular/solve.csv") == sha(fold / "regular/reload.csv")
            and seed_control["source"] == sha(fold / "regular/sealed.json") and seed_control["predictions"]["always"] == sha(replica / "always.npy")
            and read(fold / "inputs.json")["entry_queries"] == sha(fold / "anonymous/queries.zip"), "Unbound paired controls")
    payload = joblib.load(src / "refit_data.joblib")
    train_x, labels = payload["X_train"], np.asarray(payload["y_train"])
    classes = np.unique(labels)
    queries, = arrays(fold / "anonymous/queries.zip", ["C_test.npy"])
    weights = weights_for(train_x)
    save(root / "launch.json", dict(protocol=sha(PLAN), control=sha(fold / "regular/sealed.json"),
         replicas=sha(replica / "sealed.json"), labels=len(labels), queries=len(queries), input=sha(src / "refit_data.joblib"),
         seeds=list(SEEDS), weight_sd=float(weights[0].std())))
    old, standard, proposed = [], [], []
    details = []
    for j, name in enumerate(control["members"]):
        model = joblib.load(src / f"refit_component_{j}.joblib")["model"]
        p = probabilities(model, queries, classes)
        old.append(p)
        uniform = [p]
        alternatives = [p]
        for seed, row_weights in zip(SEEDS, weights, strict=True):
            candidate = reseed(model, seed)
            saved = replica / f"full_{j}_{seed}.npy"
            require((candidate is not None) == saved.is_file(), "Eligibility changed from same-cost control")
            if candidate is None:
                continue
            require(sha(saved) == read(saved.with_suffix(".json"))["probabilities"], "Standard comparator changed")
            uniform.append(np.load(saved, allow_pickle=False))
            tag = f"member_{j}_seed_{seed}"
            save(root / (tag + "_attempt.json"), dict(member=name, seed=seed, requested_classifier_fits=1))
            start = time.monotonic()
            kw = {candidate.steps[-1][0] + "__sample_weight": row_weights} if isinstance(candidate, Pipeline) else {"sample_weight": row_weights}
            candidate.fit(train_x, labels, **kw)
            result = probabilities(candidate, queries, classes)
            require(result.shape == p.shape and np.isfinite(result).all() and np.allclose(result.sum(1),1.,atol=1e-5),
                    "Invalid weighted class predictions")
            path = root / (tag + ".npy")
            np.save(path, result, allow_pickle=False)
            report = dict(member=name, seed=seed, actual_classifier_fits=1, elapsed_seconds=round(time.monotonic()-start,2),
                          predictions=sha(path))
            save(root / (tag + ".json"), report)
            details.append(report)
            alternatives.append(result)
        standard.append(np.mean(uniform, axis=0))
        proposed.append(np.mean(alternatives, axis=0))
    def hard(parts):
        return classes[np.mean(parts, axis=0).argmax(1)].astype(str)
    require(np.array_equal(hard(old), prediction_csv(fold / "regular/solve.csv", len(queries)))
            and np.array_equal(hard(standard), np.load(replica / "always.npy", allow_pickle=False)), "Failed to reproduce controls")
    target = root / "predictions.npy"
    np.save(target, hard(proposed), allow_pickle=False)
    require(len(details) == seed_control["additional_full_fits"], "Alternative has a different fit count")
    save(root / "sealed.json", dict(protocol=sha(PLAN), launch=sha(root / "launch.json"), members=control["members"],
         details=details, actual_fits=len(details), rows=len(queries), predictions=sha(target), api_requests=0))
    print(i, len(details), flush=True)


def grade(out):
    root = out / "paired_weights"
    locks = {str(i):sha(root / str(i) / "sealed.json") for i in range(5)}
    save(root / "score_lock.json", dict(protocol=sha(PLAN), receipts=locks))
    hits = {name:[] for name in ("original", "uniform", "weighted")}
    records, groups = [], []
    for i in range(5):
        fold, area = get_fold(out, 48, i), root / str(i)
        require(read(area / "sealed.json")["predictions"] == sha(area / "predictions.npy")
                and sha(fold / "outer_answers.joblib") == read(fold / "inputs.json")["outer_answers"], "Wrong answers or result")
        answer = joblib.load(fold / "outer_answers.joblib")
        y = answer["y"].astype(str)
        groups.extend(answer["groups"])
        scores = {}
        for key, pred in (("original", prediction_csv(fold / "regular/solve.csv", len(y))),
                          ("uniform", np.load(out / "seed_replication" / str(i) / "always.npy", allow_pickle=False)),
                          ("weighted", np.load(area / "predictions.npy", allow_pickle=False))):
            correct = pred == y
            hits[key].extend(correct)
            scores[key] = int(correct.sum())
        records.append(dict(fold=i, rows=len(y), correct=scores, fits=read(area / "sealed.json")["actual_fits"]))
    paired = {k:compare(np.asarray(hits["weighted"]), np.asarray(hits[k]), groups) for k in ("original","uniform")}
    delta = [(f["correct"]["weighted"] - f["correct"]["uniform"]) / f["rows"] for f in records]
    result = dict(lock=sha(root / "score_lock.json"), rows=len(groups), folds=records, correct={k:int(np.sum(v)) for k,v in hits.items()},
                  versus=paired, fits=sum(r["fits"] for r in records), api_requests=0,
                  advance=bool(paired["uniform"]["gain"] >= .003 and sum(d>0 for d in delta) >= 3 and min(delta) >= -.003))
    save(root / "result.json", result)
    print(result["correct"], paired, result["advance"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    if args.action == "train":
        for idx in range(5):
            train(args.out, idx)
    else:
        grade(args.out)
