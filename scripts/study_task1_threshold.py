"""Replay a train-only decision threshold for the winner of ten frozen autonomous runs."""
import argparse
import gc
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.score_evidence import checked_score_record

PLAN = ROOT / "research/task1_feedback_extension_1127/decision_threshold.json"
GRID = np.arange(380, 621, 15) / 1000.
SPLITS = ("primary", "confirmation_1", "confirmation_2")


def choose(folds):
    if len(folds) != 3:
        return .5, []
    counts = np.array([np.count_nonzero((p[:, None] > GRID) == y[:, None], axis=0) for p, y in folds])
    records = []
    neutral = list(GRID).index(.5)
    for i, (p, _) in enumerate(folds):
        held = counts[np.arange(3) != i].sum(0)
        chosen = min(range(len(GRID)), key=lambda j: (-held[j], abs(GRID[j] - .5), GRID[j]))
        records.append(dict(cutoff=float(GRID[chosen]), held_rows=len(p), net=int(counts[i, chosen] - counts[i, neutral])))
    cutoffs = np.array([r["cutoff"] for r in records])
    median = float(np.median(cutoffs))
    changes = np.array([r["net"] for r in records])
    sizes = np.array([r["held_rows"] for r in records])
    accepted = (max(sum(cutoffs > .5), sum(cutoffs < .5)) >= 2 and median != .5
                and min(changes / sizes) >= -.003 and changes.sum() / sizes.sum() >= .003)
    return median if accepted else .5, records


def get_fold(out, cap, index):
    if cap == 48:
        return out / ("remaining_folds" if index < 3 else "ordinary_catalogue") / str(index)
    return out / ("fusion_match" if index < 3 else "near_full") / str(index)


def positive(p):
    return p[:, 1] / p.sum(axis=1)


def run(out):
    root = out / "threshold_replay"
    root.mkdir(exist_ok=False)
    locks = {}
    for cap in (32, 48):
        for i in range(5):
            fold = get_fold(out, cap, i)
            side = fold / "regular"
            native = side / "run/agent"
            receipt = read(side / "sealed.json")
            info = read(fold / "inputs.json")
            require(receipt["independent_reload"] and receipt["prediction"] == sha(side / "solve.csv") == sha(side / "reload.csv")
                    and read(side / "launch.json")["cap"] == cap, "Unverified baseline")
            members = receipt["members"]
            prior = prediction_csv(side / "solve.csv", info["roles"]["external"]["rows"])
            files = {(r["candidate"], r["split_id"]): native / f"trial_{r['trial']}.joblib"
                     for r in read(native / "history.json") if r["status"] == "completed"}
            folds = []
            classes = None
            for split in SPLITS:
                if any((m, split) not in files for m in members):
                    continue
                data = native / (split + ".joblib")
                payload = joblib.load(data)
                metric = MetricSpec.from_payload(payload)
                new_classes = np.asarray(metric.labels)
                require(len(new_classes) == 2 and (classes is None or np.array_equal(classes, new_classes)),
                        "Unexpected output classes")
                classes = new_classes
                y = payload["y_search"]
                matrix = []
                for member in members:
                    bundle = joblib.load(files[member, split])
                    record = bundle["prediction_record"]
                    require(record["evaluation_payload_sha256"] == sha(data), "Split fingerprint mismatch")
                    p, order = checked_score_record(record["score_record"], record, metric, member, y)
                    matrix.append(p[:, [list(order).index(c) for c in classes]])
                    del bundle
                folds.append((positive(np.mean(matrix, axis=0)), np.asarray(y) == classes[1]))
            threshold, checks = choose(folds)
            directory = root / f"{cap}_{i}"
            directory.mkdir()
            save(directory / "decision.json", dict(protocol=sha(PLAN), source=sha(side / "sealed.json"),
                 members=members, folds=len(folds), cutoff=threshold, training_feedback=checks))
            if threshold == .5:
                result = prior
            else:
                query = fold / "anonymous/queries.zip"
                require(sha(query) == info["entry_queries"], "Unbound queries")
                rows, = arrays(query, ["C_test.npy"])
                parts = []
                for j, member in enumerate(members):
                    bundle = joblib.load(native / f"refit_component_{j}.joblib")
                    order, p = probability_with_model("categorical_tabular", bundle["model"], rows)
                    parts.append(p[:, [list(order).index(c) for c in classes]])
                    del bundle
                matrix = np.mean(parts, axis=0)
                require(np.array_equal(classes[matrix.argmax(1)].astype(str), prior), "Changed original decisions")
                result = classes[(positive(matrix) > threshold).astype(int)].astype(str)
            p = directory / "predictions.npy"
            np.save(p, np.asarray(result).astype(str), allow_pickle=False)
            save(directory / "sealed.json", dict(decision=sha(directory / "decision.json"), predictions=sha(p),
                 rows=len(prior), changed=int(np.sum(result != prior)), additional_base_fits=0))
            locks[directory.name] = sha(directory / "sealed.json")
            gc.collect()
            print("threshold", cap, i, threshold, flush=True)
    save(root / "lock.json", dict(protocol=sha(PLAN), predictions=locks))


def grade(out):
    root = out / "threshold_replay"
    lock = read(root / "lock.json")
    results = {}
    for cap in (32, 48):
        records, hits, baselines, groups = [], [], [], []
        for i in range(5):
            fold = get_fold(out, cap, i)
            area = root / f"{cap}_{i}"
            seal = read(area / "sealed.json")
            info = read(fold / "inputs.json")
            path = fold / "outer_answers.joblib"
            require(lock["predictions"][area.name] == sha(area / "sealed.json") and sha(area / "predictions.npy") == seal["predictions"]
                    and sha(path) == info["outer_answers"], "Unbound predictions or scoring packet")
            answer = joblib.load(path)
            y = answer["y"].astype(str)
            old = prediction_csv(fold / "regular/solve.csv", len(y)) == y
            new = np.load(area / "predictions.npy", allow_pickle=False) == y
            decision = read(area / "decision.json")
            records.append(dict(fold=i, rows=len(y), cutoff=decision["cutoff"], inner_checks=decision["training_feedback"],
                                original_correct=int(old.sum()), adjusted_correct=int(new.sum()),
                                paired=compare(new, old, answer["groups"])))
            hits.extend(new)
            baselines.extend(old)
            groups.extend(answer["groups"])
        results[str(cap)] = dict(rows=len(groups), folds=records, pooled=compare(np.asarray(hits), np.asarray(baselines), groups))
    passes = (results["48"]["pooled"]["gain"] >= .003 and results["32"]["pooled"]["gain"] > 0.
              and min(r["paired"]["gain"] for d in results.values() for r in d["folds"]) >= -.003)
    save(root / "result.json", dict(lock=sha(root / "lock.json"), results=results, advance_to_fresh_entry=passes,
         additional_classifier_fits=0, api_requests=0, official_test_opened=False))
    print({k:v["pooled"] for k,v in results.items()}, passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    (run if args.action == "run" else grade)(args.out)
