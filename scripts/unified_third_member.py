"""Screen saved candidates for errors the two delivered classifiers have in common."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from scripts.unified_confidence_study import paired_interval
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha
from superagent.ab_complement import canonical_probabilities
from superagent.data import load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import checked_score_record


def decision(columns, ties):
    mean = np.mean(columns, axis=0)
    return ((mean > .5) | ((mean == .5) & ties)).astype(int)


def same_split(agent, split, rows):
    path = agent / (split + ".joblib")
    payload = joblib.load(path)
    labels, pairs = payload["y_search"], payload["X_search"]
    digest = sha(path)
    metric = MetricSpec.from_payload(payload)
    observed = {}
    for row in rows:
        if row["split_id"] == split:
            require(row["evaluation_payload_sha256"] == digest, "Nonmatching inner partitions")
            matrix, order = checked_score_record(row["score_record"], row, metric, row["candidate_id"], labels)
            observed[row["candidate_id"]] = (matrix[:, order.index(1)], row["trial"])
    return labels, np.asarray([a < b for a, b in pairs]), observed


def counts(labels, ties, matrix, members, candidate):
    columns = [matrix[name][0] for name in members]
    correct = decision(columns, ties) == labels
    triple = decision(columns + [matrix[candidate][0]], ties) == labels
    both_wrong = np.logical_and.reduce([decision([col], ties) != labels for col in columns])
    resolves = decision([matrix[candidate][0]], ties) == labels
    return dict(pair=int(correct.sum()), triple=int(triple.sum()), net=int(triple.sum() - correct.sum()),
                shared_errors=int(both_wrong.sum()), rescued_by_candidate_alone=int((both_wrong & resolves).sum()),
                trial=matrix[candidate][1], rows=len(labels))


def select(raw, out):
    agent = raw / "titles/run/agent"
    choice = read(agent / "final_selection.json")
    members = choice["ensemble_components"]
    require(len(members) == 2 and read(raw / "titles/completion.json")["model"] == choice["model_sha256"],
            "Two frozen raw-entry members are required")
    evidence = read(agent / "prediction_evidence.json")
    labels, ties, table = same_split(agent, "primary", evidence)
    candidates = {k: counts(labels, ties, table, members, k) for k in table if k not in members}
    chosen = max(candidates, key=lambda k: (candidates[k]["net"], candidates[k]["rescued_by_candidate_alone"], -candidates[k]["trial"]))
    independent = {}
    for split in dict.fromkeys(r["split_id"] for r in evidence if r["split_id"] != "primary"):
        y2, t2, other = same_split(agent, split, evidence)
        if {*members, chosen} <= other.keys():
            independent[split] = counts(y2, t2, other, members, chosen)
    trials = {k: {"trial": table[k][1], "artifact": sha(agent / f"trial_{table[k][1]}.joblib")} for k in [*members, chosen]}
    save(out / "selection.json", dict(members=members, candidates=candidates, chosen=chosen, independent=independent,
        trial_artifacts=trials, delivered=choice["model_sha256"], study=sha(Path(__file__)),
        plan=sha(ROOT / "docs/batches/unified_third_member_protocol_20261124_zh.md")))


def predict(raw, out):
    selected = read(out / "selection.json")
    run = raw / "titles/run"
    agent = run / "agent"
    pairs = load_prediction_features(run / "audit_features.zip", "pair_text", read(agent / "task_spec.json"))
    ties = np.asarray([a < b for a, b in pairs])
    saved = {}
    for name, v in selected["trial_artifacts"].items():
        path = agent / f"trial_{v['trial']}.joblib"
        require(sha(path) == v["artifact"], "Search model changed")
        saved[name] = canonical_probabilities(joblib.load(path)["model"], pairs)[1][:, 1]
    path = agent / "selected_model.joblib"
    require(sha(path) == selected["delivered"], "Delivered model changed")
    fitted = joblib.load(path)["model"]
    third = saved[selected["chosen"]]
    columns = [saved[n] for n in selected["members"]]
    final = [canonical_probabilities(model, pairs)[1][:, 1] for model in fitted.models]
    with (run / "audit/predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["row_id"]) for r in rows] == list(range(len(pairs))) and
            np.array_equal(decision(final, ties), [int(r["prediction"]) for r in rows]), "Delivered control changed")
    values = [decision(columns, ties), decision(columns + [third], ties), decision(final, ties),
              decision(final + [third], ties), decision([third], ties), decision(final[:1], ties), decision(final[1:], ties)]
    target = out / "predictions.csv"
    with target.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "matched_pair", "matched_triple", "delivered_pair", "delivered_triple",
                         "third", "first_delivered", "second_delivered"])
        writer.writerows((i, *map(int, row)) for i, row in enumerate(zip(*values, strict=True)))
    save(out / "freeze.json", dict(selection=sha(out / "selection.json"), predictions=sha(target),
                                   features=sha(run / "audit_features.zip"), rows=len(pairs)))


def assess(labels, predictions, groups):
    control, challenge = (x == labels for x in predictions)
    sizes = np.bincount(groups)
    deltas = np.bincount(groups, weights=challenge.astype(int) - control.astype(int)).astype(int)
    interval = paired_interval(sizes, deltas)
    gain, lost = int((challenge & ~control).sum()), int((control & ~challenge).sum())
    return dict(pair_correct=int(control.sum()), triple_correct=int(challenge.sum()), gained=gain, lost=lost,
                interval_fraction=interval, passes=bool(gain - lost >= 5 and interval[0] > 0))


def score(raw, study, out):
    seal, selection = read(out / "freeze.json"), read(out / "selection.json")
    require(sha(out / "predictions.csv") == seal["predictions"] and sha(out / "selection.json") == seal["selection"] and
            sha(study / "development_truth.json") == read(study / "protocol.json")["development_truth"] and
            read(raw / "titles/completion.json")["training"]["audit_row_ids_sha256"] == fingerprint(read(EVIDENCE / "title_reserve_ids.json")),
            "Unmatched frozen outputs or training reserve")
    data = read(study / "development_truth.json")
    y = np.asarray(data["labels"])
    _, groups = np.unique(data["groups"], return_inverse=True)
    with (out / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["row_id"]) for r in rows] == list(range(len(y))), "Mismatched training row order")
    values = {k: np.asarray([int(row[k]) for row in rows]) for k in rows[0]}
    matched = assess(y, (values["matched_pair"], values["matched_triple"]), groups)
    delivered = assess(y, (values["delivered_pair"], values["delivered_triple"]), groups)
    errors = (values["first_delivered"] != y) & (values["second_delivered"] != y)
    require(delivered["pair_correct"] == read(raw / "titles/run/audit.json")["correct"], "Control did not reproduce")
    independent = selection["independent"]
    result = dict(chosen=selection["chosen"], training_rows=len(y), independent_groups=len(np.unique(groups)),
                  delivered_common_errors=int(errors.sum()), corrected_by_third_alone=int((errors & (values["third"] == y)).sum()),
                  corrected_by_actual_triple=int((errors & (values["delivered_triple"] == y)).sum()), matched=matched, delivered=delivered,
                  refit_gate_passed=bool(selection["candidates"][selection["chosen"]]["net"] > 0 and independent
                     and all(v["net"] > 0 for v in independent.values()) and matched["passes"] and delivered["passes"]),
                  model_api_calls=0, additional_fits=0, course_validation_used=False, course_test_used=False)
    save(out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw", "study", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    select(a.raw, a.out)
    predict(a.raw, a.out)
    score(a.raw, a.study, a.out)
