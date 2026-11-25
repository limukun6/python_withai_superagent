"""Evaluate a percentile-based tie breaker for two independently scored preference models."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha
from superagent.ab_complement import canonical_probabilities
from superagent.data import load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import checked_score_record


def percentiles(history, values):
    reference = np.sort(np.abs(np.asarray(history) - .5))
    distances = np.abs(np.asarray(values) - .5)
    return (np.searchsorted(reference, distances, side="left") +
            np.searchsorted(reference, distances, side="right")) / (2 * len(reference))


def paired_interval(sizes, deltas):
    rng = np.random.default_rng(28113)
    estimates = []
    for _ in range(2000):
        sample = rng.integers(0, len(sizes), size=len(sizes))
        estimates.append(deltas[sample].sum() / sizes[sample].sum())
    return np.quantile(estimates, [.025, .975]).tolist()


def generate(raw, out):
    run = raw / "titles/run"
    agent = run / "agent"
    sel = read(agent / "final_selection.json")
    members = sel["ensemble_components"]
    require((raw / "results.json").is_file() and len(members) == 2 and
            sha(agent / "selected_model.joblib") == sel["model_sha256"], "Two frozen raw-entry members required")
    evidence = read(agent / "prediction_evidence.json")
    ordered = list(dict.fromkeys(row["split_id"] for row in evidence if row["split_id"] != "primary"))
    split = next(s for s in ordered if set(members) <= {r["candidate_id"] for r in evidence if r["split_id"] == s})
    payload = agent / (split + ".joblib")
    data = joblib.load(payload)
    metric = MetricSpec.from_payload(data)
    pairs = load_prediction_features(run / "audit_features.zip", "pair_text", read(agent / "task_spec.json"))
    model = joblib.load(agent / "selected_model.joblib")["model"]
    tie = np.array([a < b for a, b in pairs])
    choices, ranks = [], []
    for name, member in zip(members, model.models, strict=True):
        row = next(r for r in evidence if r["candidate_id"] == name and r["split_id"] == split)
        require(row["evaluation_payload_sha256"] == sha(payload), "Unmatched score reference")
        scores, labels = checked_score_record(row["score_record"], row, metric, name, data["y_search"])
        probability = canonical_probabilities(member, pairs)[1][:, 1]
        choices.append((probability > .5) | ((probability == .5) & tie))
        ranks.append(percentiles(scores[:, labels.index(1)], probability))
    with (run / "audit/predictions.csv").open() as stream:
        base_rows = list(csv.DictReader(stream))
    require([int(r["row_id"]) for r in base_rows] == list(range(len(pairs))), "Unaligned baseline")
    base = np.array([int(r["prediction"]) for r in base_rows])
    disagreement = choices[0] != choices[1]
    require(np.all(base[~disagreement] == choices[0][~disagreement]), "Agreements differ from frozen baseline")
    challenge = np.where(disagreement & (ranks[0] > ranks[1]), choices[0],
                         np.where(disagreement & (ranks[1] > ranks[0]), choices[1], base)).astype(int)
    target = out / "predictions.csv"
    with target.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "control", "percentile", "disagree"])
        writer.writerows((i, int(x), int(y), int(z)) for i, (x, y, z) in enumerate(zip(base, challenge, disagreement)))
    save(out / "freeze.json", dict(predictions=sha(target), model=sel["model_sha256"],
         reference_split=split, reference_rows=len(data["y_search"]), raw_audit_input=sha(run / "audit_features.zip"),
         raw_entry=sha(ROOT / "unified_agent.py"), plan=sha(ROOT / "docs/batches/unified_confidence_protocol_20261124_zh.md"),
         runner=sha(Path(__file__)), members=members))


def score(raw, study, out):
    seal = read(out / "freeze.json")
    require(sha(out / "predictions.csv") == seal["predictions"] and
            sha(study / "development_truth.json") == read(study / "protocol.json")["development_truth"],
            "Sealed input changed")
    with (out / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    data = read(study / "development_truth.json")
    y = np.array(data["labels"])
    reserve_ids = EVIDENCE / "title_reserve_ids.json"
    require(sha(reserve_ids) == read(study / "protocol.json")["reserve_ids"] and
            read(raw / "titles/completion.json")["training"]["audit_row_ids_sha256"] == fingerprint(read(reserve_ids)),
            "Different training reserves")
    control = np.array([int(r["control"]) for r in rows]) == y
    new = np.array([int(r["percentile"]) for r in rows]) == y
    audit = read(raw / "titles/run/audit.json")
    require(int(control.sum()) == audit["correct"] and len(rows) == audit["rows"], "Baseline does not replay")
    _, ids = np.unique(data["groups"], return_inverse=True)
    sizes = np.bincount(ids)
    deltas = np.bincount(ids, weights=new.astype(int) - control.astype(int)).astype(int)
    interval = paired_interval(sizes, deltas)
    gained, lost = int((new & ~control).sum()), int((control & ~new).sum())
    disagree = np.asarray([bool(int(r["disagree"])) for r in rows])
    shared_errors = int((~control & ~disagree).sum())
    save(out / "groups.json", dict(rows=sizes.tolist(), differences=deltas.tolist()))
    result = dict(rows=len(rows), groups=len(sizes),
                  control_correct=int(control.sum()), candidate_correct=int(new.sum()),
                  member_disagreements=int(disagree.sum()), errors_both_members_share=shared_errors,
                  control_errors_on_disagreements=int((~control & disagree).sum()), gained=gained, lost=lost,
                  interval_fraction=interval, passed=bool(gained - lost >= 5 and interval[0] > 0),
                  fits=0, model_api_calls=0, designated_validation_used=False, official_test_used=False)
    save(out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "study", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    generate(a.raw, a.out)
    score(a.raw, a.study, a.out)
