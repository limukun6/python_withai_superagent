"""Repeat only seedable selected classifiers on identical folds and let training feedback accept or retain."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_threshold import get_fold, SPLITS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import ContextTokenTree
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.score_evidence import checked_score_record

PLAN = ROOT / "research/task1_feedback_extension_1127/full_training_seed_replication.json"
SEEDS = (13291, 94723)


def reseed(model, seed):
    if isinstance(model, ContextTokenTree):
        params = deepcopy(model.params)
        params["tree_params"] = {**params.get("tree_params", {}), "random_seed": seed}
        return type(model)(params)
    if isinstance(model, Pipeline):
        name, estimator = model.steps[-1]
        if "random_state" in estimator.get_params(deep=False):
            return clone(model).set_params(**{name + "__random_state": seed})
    return None


def probabilities(model, rows, classes):
    order, p = probability_with_model("categorical_tabular", model, rows)
    return p[:, [list(order).index(c) for c in classes]]


def ensemble(area, tag, model, train, y, query, classes, original):
    parts = [original]
    for seed in SEEDS:
        copy = reseed(model, seed)
        if copy is None:
            break
        key = f"{tag}_{seed}"
        prediction = area / (key + ".npy")
        receipt = area / (key + ".json")
        if receipt.is_file():
            sealed = read(receipt)
            require(sha(prediction) == sealed["probabilities"] and sealed["rows"] == len(query)
                    and sealed["launch"] == sha(area / "launch.json"), "Changed saved replica")
            parts.append(np.load(prediction, allow_pickle=False))
            continue
        attempt = area / (key + "_attempt.json")
        require(not attempt.exists(), "Interrupted replica requires separate accounting")
        save(attempt, dict(launch=sha(area / "launch.json"), seed=seed, classifier=type(copy).__name__, rows=len(y)))
        copy.fit(train, y)
        p = probabilities(copy, query, classes)
        require(p.shape == original.shape and np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5),
                "Invalid replica probabilities")
        np.save(prediction, p, allow_pickle=False)
        entry = dict(launch=sha(area / "launch.json"), attempt=sha(attempt), probabilities=sha(prediction),
                     rows=len(query), actual_fits=1)
        if tag.startswith("full_"):
            model_file = area / (key + ".joblib")
            joblib.dump(copy, model_file, compress=3)
            entry["model"] = sha(model_file)
        save(receipt, entry)
        parts.append(p)
        del copy
    return np.mean(parts, axis=0)


def run(out, idx):
    fold = get_fold(out, 48, idx)
    side = fold / "regular"
    src = side / "run/agent"
    control = read(side / "sealed.json")
    require(control["independent_reload"] and control["prediction"] == sha(side / "solve.csv") == sha(side / "reload.csv"),
            "Unverified baseline")
    root = out / "seed_replication"
    root.mkdir(exist_ok=True)
    area = root / str(idx)
    area.mkdir(exist_ok=True)
    save(area / "launch.json", dict(protocol=sha(PLAN), source=sha(side / "sealed.json"), owner=sha(fold / "inputs.json"),
                                    members=control["members"], base_fits=control["fits"], seeds=list(SEEDS)))
    history = {(r["candidate"], r["split_id"]): r["trial"] for r in read(src / "history.json") if r["status"] == "completed"}
    members = control["members"]
    feedback = []
    for split in SPLITS:
        if not all((name, split) in history for name in members):
            continue
        data = src / (split + ".joblib")
        payload = joblib.load(data)
        metadata = read(src / (split + "_split.json"))
        require(sha(data) == metadata["payload_sha256"], "Changed feedback data")
        metric = MetricSpec.from_payload(payload)
        classes = np.asarray(metric.labels)
        old, new = [], []
        y = np.asarray(payload["y_search"])
        for j, name in enumerate(members):
            bundle = joblib.load(src / f"trial_{history[name, split]}.joblib")
            record = bundle["prediction_record"]
            require(record["evaluation_payload_sha256"] == metadata["payload_sha256"], "Unbound score evidence")
            probs, order = checked_score_record(record["score_record"], record, metric, name, y)
            p = probs[:, [list(order).index(c) for c in classes]]
            old.append(p)
            new.append(ensemble(area, f"{split}_{j}", bundle["model"], payload["X_train"], payload["y_train"],
                                payload["X_search"], classes, p))
            del bundle
        a, b = np.mean(old, axis=0), np.mean(new, axis=0)
        diff = (classes[b.argmax(1)] == y).astype(int) - (classes[a.argmax(1)] == y).astype(int)
        feedback.append(dict(split=split, rows=len(y), net=int(diff.sum()), gain=float(diff.mean()),
                             log_loss_difference=float(log_loss(y, b, labels=classes) - log_loss(y, a, labels=classes))))
        gc.collect()
    total = sum(r["rows"] for r in feedback)
    accepted = (len(feedback) == 3 and sum(r["net"] >= 0 for r in feedback) >= 2
                and min(r["gain"] for r in feedback) >= -.003 and sum(r["net"] for r in feedback) / total >= .001
                and sum(r["rows"] * r["log_loss_difference"] for r in feedback) <= 0)
    save(area / "decision.json", dict(protocol=sha(PLAN), use_replicas=accepted, feedback=feedback))
    train_path = src / "refit_data.joblib"
    payload = joblib.load(train_path)
    rows, = arrays(fold / "anonymous/queries.zip", ["C_test.npy"])
    classes = np.asarray(MetricSpec.from_payload(payload).labels)
    old, new = [], []
    for j, name in enumerate(members):
        bundle = joblib.load(src / f"refit_component_{j}.joblib")
        p = probabilities(bundle["model"], rows, classes)
        old.append(p)
        new.append(ensemble(area, f"full_{j}", bundle["model"], payload["X_train"], payload["y_train"],
                            rows, classes, p))
        del bundle
    primary = classes[np.mean(old, axis=0).argmax(1)].astype(str)
    repeated = classes[np.mean(new, axis=0).argmax(1)].astype(str)
    require(np.array_equal(primary, prediction_csv(side / "solve.csv", len(rows))), "Wrong selected baseline")
    signatures = {}
    for key, result in (("decision", repeated if accepted else primary), ("always", repeated)):
        path = area / (key + ".npy")
        np.save(path, result, allow_pickle=False)
        signatures[key] = sha(path)
    completed = [r for r in area.glob("*.json") if "actual_fits" in read(r)]
    fits = sum(read(path)["actual_fits"] for path in completed)
    final_fits = sum(read(p)["actual_fits"] for p in completed if p.name.startswith("full_"))
    require(fits == len(list(area.glob("*_attempt.json"))) and fits + control["fits"] <= 64, "Unexpected physical fit cost")
    save(area / "sealed.json", dict(protocol=sha(PLAN), decision=sha(area / "decision.json"), source=sha(side / "sealed.json"),
         rows=len(rows), predictions=signatures, extra_fits=fits, additional_check_fits=fits-final_fits,
         additional_full_fits=final_fits, simulated_path_fits=control["fits"] + fits - (0 if accepted else final_fits),
         total_fits_including_diagnostic=fits + control["fits"], api_requests=0))
    print(idx, accepted, feedback, "extra fits", fits, flush=True)


def grade(out):
    root = out / "seed_replication"
    seals = {str(i): sha(root / str(i) / "sealed.json") for i in range(5)}
    save(root / "scoring_lock.json", dict(protocol=sha(PLAN), folds=seals))
    hits = {k:[] for k in ("partner","regular","decision","always")}
    all_groups, records = [], []
    for idx in range(5):
        fold = get_fold(out, 48, idx)
        area = root / str(idx)
        p = fold / "outer_answers.joblib"
        require(sha(p) == read(fold / "inputs.json")["outer_answers"], "Wrong outer score packet")
        answer = joblib.load(p)
        y = answer["y"].astype(str)
        seal = read(area / "sealed.json")
        dp = read(area / "decision.json")
        scores = {}
        for name in hits:
            if name in ("partner", "regular"):
                original = fold / name
                path = original / "solve.csv"
                require(sha(path) == read(original / "sealed.json")["prediction"], "Changed control")
                labels = prediction_csv(path, len(y))
            else:
                path = area / (name + ".npy")
                require(sha(path) == seal["predictions"][name], "Changed replica predictions")
                labels = np.load(path, allow_pickle=False)
            scores[name] = labels == y
            hits[name].extend(scores[name])
        all_groups.extend(answer["groups"])
        records.append(dict(fold=idx, rows=len(y), use_replicas=dp["use_replicas"], inner=dp["feedback"],
                            extra_fits=seal["extra_fits"], simulated_path_fits=seal["simulated_path_fits"],
                            correct={k:int(v.sum()) for k,v in scores.items()},
                            gain_vs_regular=float(scores["decision"].mean() - scores["regular"].mean())))
    differences = {n:{b:compare(np.asarray(hits[n]), np.asarray(hits[b]), all_groups) for b in ("regular","partner")}
                   for n in ("decision","always")}
    changes = [r["gain_vs_regular"] for r in records]
    passes = min(d["gain"] for d in differences["decision"].values()) >= .003 and sum(c > 0 for c in changes) >= 3 and min(changes) >= -.004
    result = dict(lock=sha(root / "scoring_lock.json"), folds=records, differences=differences,
                  rows=len(all_groups), correct={k:int(np.sum(v)) for k,v in hits.items()},
                  new_fits=sum(r["extra_fits"] for r in records), qualify_fresh_entry=passes, api_requests=0)
    save(root / "result.json", result)
    print(result["correct"], result["new_fits"], passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.action == "run":
        run(args.out, args.fold)
    else:
        grade(args.out)
