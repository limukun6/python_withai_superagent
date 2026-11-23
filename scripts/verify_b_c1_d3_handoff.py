"""Check the published frozen receipts and recompute the independent-review summary using stdlib."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "research"
C1 = ROOT / "b_c1_validation_1123"
D3 = ROOT / "b_integrated_offline_1123"
BLIND = ROOT / "a70_independent_review" / "blinded_reviews.json"
ARMS = {"L0", "L1", "R0", "R1", "E0", "E1"}
FIELDS = ("factual_error", "scope_or_reference_error", "unsupported_causal_certainty", "appropriate_uncertainty")


def read(path):
    return json.loads(path.read_bytes())


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def check_c1():
    resources = read(C1 / "resource_summary.json")["scopes"]
    total_fits = total_calls = 0
    for task in ("preference", "bean"):
        base = C1 / task
        results = read(base / "results.json")
        complete = results["arms"]
        partial = results.get("partial_delivery", {})
        check(set(partial) == ({"R0", "E1"} if task == "bean" else set()), task)
        check(not set(complete).intersection(partial) and set(complete) | set(partial) == ARMS, task)
        seals = read(base / "all_six_terminal_seals.json")
        predictions = read(base / "all_six_predictions_frozen.json")["export_sha256"]
        reloads = read(base / "rechecks.json")
        for arm, row in {**complete, **partial}.items():
            label = f"{task}/{arm}"
            check(all(row[k] == v for k, v in seals[arm].items()), label)
            check(row["prediction_sha256"] == predictions[arm], label)
            check(row["model_sha256"] == reloads[arm]["model_sha256"], label)
            check(row["rows"] == reloads[arm]["rows"] and row["accuracy"] == reloads[arm]["accuracy"], label)
            check(row["correct"] / row["rows"] == row["accuracy"], label)
        rows = [*complete.values(), *partial.values()]
        scope = resources[task]
        check(sum(r["fits"] for r in rows) == scope["fits"], task)
        check(sum(r["requests"] for r in rows) == scope["requests"], task)
        total_fits += scope["fits"]
        total_calls += scope["requests"]
    return {"models": 12, "completed_planners": 10, "historical_fits": total_fits, "historical_calls": total_calls}


def check_d3():
    frozen = read(D3 / "independent_summary" / "protocol.json")["input_hashes"]
    for name, digest in frozen.items():
        check(sha256((ROOT.parent / name).read_bytes()).hexdigest() == digest, name)
    cases = read(D3 / "a1_cases.json")
    outcomes = read(D3 / "a1_check" / "results.json")
    summary = read(D3 / "independent_summary" / "results.json")
    categories = Counter(r["category"] for r in cases)
    check(len(categories) == 8 and set(categories.values()) == {6}, "case coverage")
    check(len({r["id"] for r in cases}) == len(cases) == len(outcomes) == 48, "case IDs")
    matrix = Counter()
    for case, row in zip(cases, outcomes):
        check((case["id"], case["expected"]) == (row["id"], row["expected"]), "case alignment")
        matrix[row["expected"], row["observed"]] += 1
    for expected, column in summary["a1"]["expected_by_observed"].items():
        for observed, count in column.items():
            check(matrix[expected, observed] == count, "case confusion matrix")
    reviews = read(D3 / "a2_annotated_reviews.json")
    original = read(BLIND)
    check(len({r["review_id"] for r in reviews}) == len(original) == len(reviews) == 24, "review IDs")
    ratings = {name: Counter() for name in FIELDS}
    for source, annotated in zip(original, reviews):
        tags = annotated["independent_annotation"]
        check({**source, "independent_annotation": tags} == annotated, source["review_id"])
        check(tags["checker_outputs_seen"] is False and tags["annotator"] and tags["label_basis"], source["review_id"])
        for field in FIELDS:
            value = tags[field]
            check(type(value) is bool or value is None, field)
            ratings[field][{True: "yes", False: "no", None: "unknown"}[value]] += 1
    for field in FIELDS:
        for label, n in summary["a2"]["ratings"][field].items():
            check(ratings[field][label] == n, field)
    return {"cases": len(cases), "checker_agreements": sum(v for (e, o), v in matrix.items() if e == o),
            "blind_reviews": len(reviews), "factual_error_yes": ratings["factual_error"]["yes"]}


if __name__ == "__main__":
    print(json.dumps({"c1": check_c1(), "independent": check_d3()}, indent=2))
