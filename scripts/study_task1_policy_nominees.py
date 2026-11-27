"""Evaluate whole search-policy nominees as interchangeable advisors using matched training feedback."""
import argparse
import gc
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_inference_reuse import fit_weight
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_threshold import get_fold, SPLITS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.score_evidence import checked_score_record

PLAN = ROOT / "research/task1_feedback_extension_1127/policy_nominee_blending.json"


def internal(area, split, members, history):
    data = area / (split + ".joblib")
    meta = read(area / (split + "_split.json"))
    require(sha(data) == meta["payload_sha256"], "Unbound training fold")
    payload = joblib.load(data)
    y = payload["y_search"]
    metric = MetricSpec.from_payload(payload)
    classes = list(metric.labels)
    components = []
    for member in members:
        bundle = joblib.load(area / f"trial_{history[member, split]}.joblib")
        record = bundle["prediction_record"]
        require(record["evaluation_payload_sha256"] == meta["payload_sha256"], "Wrong scoring panel")
        p, order = checked_score_record(record["score_record"], record, metric, member, y)
        components.append(p[:, [list(order).index(k) for k in classes]])
        del bundle
    return meta, metric, y, np.mean(components, axis=0)


def run(out):
    obj = out / "policy_nominees"
    obj.mkdir(exist_ok=False)
    locks = {}
    for cap in (32, 48):
        for idx in range(5):
            fold = get_fold(out, cap, idx)
            info = read(fold / "inputs.json")
            names = ("prior" if cap == 32 else "partner", "regular")
            sides = [fold / n for n in names]
            manifests = [read(s / "sealed.json") for s in sides]
            histories = []
            existing = []
            for s, d in zip(sides, manifests, strict=True):
                require(d["independent_reload"] and d["prediction"] == sha(s / "solve.csv") == sha(s / "reload.csv")
                        and read(s / "launch.json")["inputs"] == sha(fold / "inputs.json"), "Changed election")
                histories.append({(r["candidate"], r["split_id"]):r["trial"]
                                  for r in read(s / "run/agent/history.json") if r["status"] == "completed"})
                existing.append(prediction_csv(s / "solve.csv", info["roles"]["external"]["rows"]))
            area = obj / f"{cap}_{idx}"
            area.mkdir()
            scoring = []
            classes = None
            for split in SPLITS:
                if not all(all((m, split) in h for m in d["members"]) for h, d in zip(histories, manifests)):
                    continue
                records = [internal(s / "run/agent", split, d["members"], h) for s,d,h in zip(sides, manifests, histories)]
                a, b = records
                require(a[0]["row_sha256"] == b[0]["row_sha256"] and a[0]["label_sha256"] == b[0]["label_sha256"]
                        and a[0]["held"] == b[0]["held"] and a[1].labels == b[1].labels
                        and np.array_equal(a[2], b[2]), "Mismatched advisor rows or labels")
                classes = np.asarray(a[1].labels)
                metric = a[1]
                scoring.append((a[2], [a[3], b[3]]))
            require(classes is not None, "Both nominees need at least one common feedback panel")
            weight, checks = fit_weight(scoring, metric, classes)
            save(area / "decision.json", dict(protocol=sha(PLAN), shares=[weight, 1. - weight],
                 folds=len(scoring), checks=checks, member_lists=[d["members"] for d in manifests],
                 actual_fits=[d["fits"] for d in manifests], seals=[sha(s / "sealed.json") for s in sides]))
            if np.array_equal(*existing) or weight in (0., 1.):
                choice = existing[0 if weight != 0. else 1]
            else:
                q = fold / "anonymous/queries.zip"
                require(sha(q) == info["entry_queries"], "Changed query")
                rows, = arrays(q, ["C_test.npy"])
                proposals = []
                for s, d, prior in zip(sides, manifests, existing):
                    pieces = []
                    for j, _ in enumerate(d["members"]):
                        bundle = joblib.load(s / "run/agent" / f"refit_component_{j}.joblib")
                        order, p = probability_with_model("categorical_tabular", bundle["model"], rows)
                        pieces.append(p[:, [list(order).index(k) for k in classes]])
                        del bundle
                    proposal = np.mean(pieces, axis=0)
                    require(np.array_equal(prior, classes[proposal.argmax(1)].astype(str)), "Original hard labels changed")
                    proposals.append(proposal)
                choice = classes[(weight * proposals[0] + (1. - weight) * proposals[1]).argmax(1)].astype(str)
            path = area / "predictions.npy"
            np.save(path, choice.astype(str), allow_pickle=False)
            save(area / "sealed.json", dict(decision=sha(area / "decision.json"), predictions=sha(path),
                 changed=[int((choice != p).sum()) for p in existing], new_base_fits=0))
            locks[area.name] = sha(area / "sealed.json")
            gc.collect()
            print(cap, idx, weight, checks, flush=True)
    save(obj / "lock.json", dict(protocol=sha(PLAN), predictions=locks))


def grade(out):
    obj = out / "policy_nominees"
    lock = read(obj / "lock.json")
    results = {}
    for cap in (32, 48):
        records, groups = [], []
        hits = {k:[] for k in ("partner", "regular", "blend")}
        for idx in range(5):
            fold = get_fold(out, cap, idx)
            folder = obj / f"{cap}_{idx}"
            seal = read(folder / "sealed.json")
            p = fold / "outer_answers.joblib"
            require(lock["predictions"][folder.name] == sha(folder / "sealed.json")
                    and sha(folder / "predictions.npy") == seal["predictions"] and sha(p) == read(fold / "inputs.json")["outer_answers"],
                    "Unbound final scoring")
            answer = joblib.load(p)
            y = answer["y"].astype(str)
            old = "prior" if cap == 32 else "partner"
            by = dict(partner=prediction_csv(fold / old / "solve.csv", len(y)) == y,
                      regular=prediction_csv(fold / "regular/solve.csv", len(y)) == y,
                      blend=np.load(folder / "predictions.npy", allow_pickle=False) == y)
            for k,v in by.items():
                hits[k].extend(v)
            groups.extend(answer["groups"])
            records.append(dict(fold=idx, decision=read(folder / "decision.json"), rows=len(y),
                 correct={k:int(v.sum()) for k,v in by.items()},
                 delta={k:float(by["blend"].mean()-by[k].mean()) for k in ("partner","regular")}))
        totals = {k:compare(np.array(hits["blend"]), np.array(hits[k]), groups) for k in ("partner","regular")}
        results[str(cap)] = dict(folds=records, rows=len(groups), total_correct={k:int(np.sum(v)) for k,v in hits.items()}, versus=totals)
    passes = (min(v["gain"] for v in results["48"]["versus"].values()) >= .003
              and min(v["gain"] for v in results["32"]["versus"].values()) >= 0.
              and min(min(f["delta"].values()) for d in results.values() for f in d["folds"]) >= -.005)
    save(obj / "result.json", dict(lock=sha(obj / "lock.json"), results=results, advance_to_equal_budget_entry=passes,
         additional_classifier_fits=0, api_requests=0, official_test_opened=False))
    print({k:v["total_correct"] for k,v in results.items()}, passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    (run if args.action == "run" else grade)(args.out)
