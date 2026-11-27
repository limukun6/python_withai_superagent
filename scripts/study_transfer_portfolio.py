"""Screen optional new sentence encoders on training groups against a frozen two-model vote."""
import argparse
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

from scripts.cache_paired_encoders import configure_weights
from scripts.confirm_unused_training_groups import measure
from scripts.study_weighted_evidence import FOLDS, available, correctness, matrices, recipe
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.metrics import MetricSpec
from superagent.paired_encoder import bank_name

PLAN = ROOT / "research/paired_encoder_search_1127/transfer_protocol.json"
NAMES = ("bge_base_paired_linear", "distilroberta_paired_linear")


def lock(path, value):
    if path.is_file():
        require(read(path) == value, "A previously locked decision differs")
    else:
        save(path, value)


def run(root, weights):
    out = root / "transfer-screen"
    out.mkdir(exist_ok=True)
    agent = root / "raw/race/run/agent"
    inherited = available(root)
    incumbent = read(root / "scale-screen/results.json")["candidate_locked"]
    require(len(incumbent) == 2, "Expected the registered two-model reference")
    for name in NAMES:
        variant = name.removesuffix("_paired_linear")
        require((root / "feature_bank" / bank_name(variant)).is_file(), "Missing training-only vectors for " + variant)
    held = [set(read(agent / (fold + "_split.json"))["held"]) for fold in FOLDS]
    require(all(a.isdisjoint(b) for a, b in combinations(held, 2)), "Held-out partitions overlap")
    configure_weights(weights)
    physical = out / "physical.jsonl"
    settings = env(weights, physical)
    settings.update({PREFERENCE_CATALOG_ENV: "all-installed", "SUPERAGENT_PAIRED_FEATURE_CACHE": str(root / "feature_bank")})
    controls = recipe(incumbent)

    def evaluate(fold, options):
        source = agent / (fold + ".joblib")
        bank, y, classes, _ = matrices(agent, fold, incumbent, inherited[fold])
        before = correctness(bank, y, classes, controls)
        metric = MetricSpec.from_payload(joblib.load(source))
        needed = sorted({m for _, r in options for m in r["members"]} - set(incumbent))
        for name in needed:
            bank[name] = np.asarray(measure(name, source, agent / "runtime_registry_0.json", out / fold / name,
                                            settings, metric, y), dtype=float)
        rows = []
        for challenger, option in options:
            right = correctness(bank, y, classes, option)
            rows.append(dict(challenger=challenger, recipe=option, fold=fold, rows=len(y), reference=int(before.sum()),
                             correct=int(right.sum()), net=int(right.sum()-before.sum()),
                             gained=int(sum(right & ~before)), lost=int(sum(~right & before))))
        return rows

    pool = []
    for name in NAMES:
        for method in (recipe([name]), recipe([*incumbent, name]), recipe([*incumbent, name], [.25, .25, .5])):
            pool.append((name, method))
    primary = evaluate(FOLDS[0], pool)
    choices = []
    for name in NAMES:
        improved = [row for row in primary if row["challenger"] == name and row["net"] > 0]
        if improved:
            choices.append(min(improved, key=lambda r: (-r["net"], len(r["recipe"]["members"]),
                                                         max(r["recipe"]["weights"]))))
    lock(out / "primary_lock.json", dict(protocol=sha(PLAN), reference=controls, measured=primary, nominated=choices))
    print("primary", [(r["challenger"], r["recipe"]["weights"], r["net"]) for r in primary], flush=True)
    confirmations = evaluate(FOLDS[1], [(r["challenger"], r["recipe"]) for r in choices]) if choices else []
    eligible = []
    for first, second in zip(choices, confirmations):
        pooled = (first["net"] + second["net"]) / (first["rows"] + second["rows"])
        if second["net"] > 0 and pooled >= .003:
            eligible.append((min(first["net"]/first["rows"], second["net"]/second["rows"]), pooled, first))
    chosen = sorted(eligible, key=lambda t: (-t[0], -t[1], len(t[2]["recipe"]["members"]), t[2]["challenger"]))
    finalist = chosen[0][2] if chosen else None
    lock(out / "confirmation_lock.json", dict(confirmed=confirmations, finalist=finalist))
    third = evaluate(FOLDS[2], [(finalist["challenger"], finalist["recipe"])])[0] if finalist else None
    fits = len(physical.read_text().splitlines())
    require(fits == len(NAMES) + len(choices) + bool(finalist) <= 6, "Unexpected physical classifier count")
    report = dict(protocol_sha256=sha(PLAN), reference=controls, primary=primary, confirmed=confirmations, third=third,
                  advance_to_autonomous_run=bool(third and third["net"] > 0), new_classifier_fits=fits,
                  model_api_requests=0, released_validation_opened=False, formal_test_opened=False, H2_opened=False)
    lock(out / "results.json", report)
    print("confirmation", [(r["challenger"], r["net"]) for r in confirmations], "third", third, "fits", fits, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "weights"):
        parser.add_argument("--"+field, type=lambda s: Path(s).resolve(), required=True)
    args = parser.parse_args()
    run(args.root, args.weights)
