"""Evaluate a frozen signed response-history correction on five previously solved training folds."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.feature_extraction import DictVectorizer

from scripts.study_task1_cross_context import ContextHistory, fit_offset
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_policy_nominees import internal
from scripts.study_task1_seed_replicas import probabilities
from scripts.study_task1_threshold import get_fold, positive, SPLITS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import tokens
from superagent.models import _clean_categories

PLAN = ROOT / "research/task1_feedback_extension_1127/cross_context_outer.json"


def corrected(prior, delta):
    adjusted = expit(logit(np.clip(prior, .001, .999)) + delta)
    return np.where(delta == 0., prior, (prior + adjusted) / 2)


def build(out, idx):
    fold = get_fold(out, 48, idx)
    area = out / "cross_context_outer" / str(idx)
    area.mkdir(parents=True, exist_ok=False)
    side = fold / "regular"
    rc = read(side / "sealed.json")
    info = read(fold / "inputs.json")
    require(rc["independent_reload"] and rc["prediction"] == sha(side / "solve.csv") == sha(side / "reload.csv")
            and info["entry_queries"] == sha(fold / "anonymous/queries.zip"), "Unbound ordinary delivery")
    src = side / "run/agent"
    replica = out / "seed_replication" / str(idx)
    seed = read(replica / "sealed.json")
    require(seed["source"] == sha(side / "sealed.json") and seed["predictions"]["always"] == sha(replica / "always.npy"),
            "Seed diagnostic uses another winner or another dataset")
    full = joblib.load(src / "refit_data.joblib")
    x, y = _clean_categories(full["X_train"]), np.asarray(full["y_train"])
    classes = np.unique(y)
    query, = arrays(fold / "anonymous/queries.zip", ["C_test.npy"])
    search = {(r["candidate"], r["split_id"]):r["trial"] for r in read(src / "history.json") if r["status"] == "completed"}
    splits = [s for s in SPLITS if all((m, s) in search for m in rc["members"])]
    rows, panel_p, target, panels = [], [], [], []
    all_held = set()
    memory_count = offset_count = 0
    reason = None
    try:
        if len(classes) != 2 or len(splits) < 2:
            raise LookupError("Binary probabilities from two disjoint training panels are required")
        for split in splits:
            meta, metric, truth, p = internal(src, split, rc["members"], search)
            fit, held = meta["fit"], meta["held"]
            require(meta["resource"]["shared_groups"] == 0 and not all_held.intersection(held)
                    and set(fit).isdisjoint(held) and len(fit) + len(held) == len(x)
                    and np.array_equal(truth, y[held]) and np.array_equal(classes, metric.labels),
                    "Unaligned or overlapping calibration panels")
            payload = joblib.load(src / (split + ".joblib"))
            require(np.array_equal(_clean_categories(payload["X_train"]), x[fit])
                    and np.array_equal(_clean_categories(payload["X_search"]), x[held])
                    and set(tokens(x[fit])).isdisjoint(tokens(x[held])), "Incorrect memory split")
            all_held.update(held)
            try:
                memory = ContextHistory(x[fit], (y[fit] == classes[1]).astype(int))
            except ValueError as err:
                raise LookupError(str(err)) from err
            features, support = memory.features(x[held])
            memory_count += 1
            rows.extend(features)
            panel_p.extend(positive(p))
            target.extend(truth == classes[1])
            panels.append(dict(split=split, rows=len(held), median_neighbours=float(np.median(support))))
        try:
            full_memory = ContextHistory(x, (y == classes[1]).astype(int))
        except ValueError as err:
            raise LookupError(str(err)) from err
        memory_count += 1
        queries, _ = full_memory.features(_clean_categories(query))
        vectorizer = DictVectorizer()
        train_design = vectorizer.fit_transform(rows)
        if train_design.shape[1] == 0:
            raise LookupError("No repeated history in the training panels")
        beta, _ = fit_offset(train_design, np.asarray(target, dtype=int), np.asarray(panel_p), positive=False)
        offset_count = 1
        delta = np.clip(vectorizer.transform(queries) @ beta, -1.25, 1.25)
    except LookupError as exc:
        reason = str(exc)
        delta = np.zeros(len(query))
    priors, repeats = [], []
    for j, _ in enumerate(rc["members"]):
        component = joblib.load(src / f"refit_component_{j}.joblib")["model"]
        original = probabilities(component, query, classes)
        versions = [original]
        for rng in (13291, 94723):
            path = replica / f"full_{j}_{rng}.npy"
            if path.exists():
                require(sha(path) == read(path.with_suffix(".json"))["probabilities"], "Changed saved repeat")
                versions.append(np.load(path, allow_pickle=False))
        require(len(versions) in (1, 3), "Incomplete existing seed comparison")
        priors.append(original)
        repeats.append(np.mean(versions, axis=0))
    origins = dict(regular=positive(np.mean(priors, axis=0)), repeated=positive(np.mean(repeats, axis=0)))
    saved = {"regular":prediction_csv(side / "solve.csv", len(query)), "repeated":np.load(replica / "always.npy", allow_pickle=False)}
    seals = {}
    for name, prior in origins.items():
        require(np.array_equal(classes[(prior > .5).astype(int)].astype(str), saved[name]), "Base replay differs")
        path = area / (name + ".npy")
        np.save(path, classes[(corrected(prior, delta) > .5).astype(int)].astype(str), allow_pickle=False)
        seals[name] = sha(path)
    save(area / "sealed.json", dict(protocol=sha(PLAN), original=sha(side / "sealed.json"), replicas=sha(replica / "sealed.json"),
         members=rc["members"], panels=panels, memory_fits=memory_count, offset_fits=offset_count,
         predictions=seals, retain_reason=reason, query_rows=len(query), api_requests=0))
    print(idx, panels, reason, flush=True)


def grade(out):
    root = out / "cross_context_outer"
    locks = {str(i):sha(root / str(i) / "sealed.json") for i in range(5)}
    save(root / "scoring_lock.json", dict(protocol=sha(PLAN), receipts=locks))
    records, groups = [], []
    hits = {name:[] for name in ("regular", "repeated", "regular_adjusted", "repeated_adjusted")}
    for i in range(5):
        fold, area = get_fold(out, 48, i), root / str(i)
        seal = read(area / "sealed.json")
        require(seal["protocol"] == sha(PLAN) and sha(fold / "outer_answers.joblib") == read(fold / "inputs.json")["outer_answers"],
                "Wrong evaluation packet")
        packet = joblib.load(fold / "outer_answers.joblib")
        truth = packet["y"].astype(str)
        groups.extend(packet["groups"])
        scores = {}
        for name in ("regular","repeated"):
            path = area / (name + ".npy")
            require(sha(path) == seal["predictions"][name], "Modified proposed decisions")
            baseline = (prediction_csv(fold / "regular/solve.csv", len(truth)) if name == "regular" else
                        np.load(out / "seed_replication" / str(i) / "always.npy", allow_pickle=False))
            for key, labels in [(name, baseline), (name + "_adjusted", np.load(path, allow_pickle=False))]:
                correct = labels == truth
                hits[key].extend(correct)
                scores[key] = int(correct.sum())
        records.append(dict(fold=i, rows=len(truth), scores=scores, offset_fits=seal["offset_fits"], memory_fits=seal["memory_fits"]))
    delta = {k:compare(np.asarray(hits[k + "_adjusted"]), np.asarray(hits[k]), groups) for k in ("regular", "repeated")}
    per = [(r["scores"]["regular_adjusted"]-r["scores"]["regular"])/r["rows"] for r in records]
    result = dict(lock=sha(root / "scoring_lock.json"), rows=len(groups), folds=records, correct={k:int(np.sum(v)) for k,v in hits.items()},
                  paired=delta, advance_to_cold_run=bool(delta["regular"]["gain"] >= .003 and sum(v>0 for v in per) >= 3 and min(per) >= -.003),
                  api_requests=0, official_test_opened=False)
    save(root / "result.json", result)
    print(result["correct"], delta, result["advance_to_cold_run"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    if args.action == "build":
        for index in range(5):
            build(args.out, index)
    else:
        grade(args.out)
