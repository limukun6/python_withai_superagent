"""Compare three locked autonomous elections on fresh group-disjoint training folds."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.study_weighted_evidence import correctness, recipe
from scripts.unified_released_study import ROOT, env, read, require, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take


PLAN = ROOT / "research/paired_encoder_search_1127/portfolio_stability_protocol.json"
REVISED = "two_encoders_with_advice"


def paired(right, baseline):
    fixed = int(np.count_nonzero(right & ~baseline))
    lost = int(np.count_nonzero(baseline & ~right))
    return dict(gained=fixed, lost=lost, net=fixed-lost)


def group_interval(right, baseline, groups, seed):
    _, indices = np.unique(groups, return_inverse=True)
    sizes = np.bincount(indices)
    deltas = np.bincount(indices, weights=right.astype(float)-baseline)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(40):
        draws = rng.integers(len(sizes), size=(100, len(sizes)))
        estimates.extend(deltas[draws].sum(axis=1) / sizes[draws].sum(axis=1))
    return np.quantile(estimates, [.025, .975]).tolist()


def run(root):
    plan = read(PLAN)
    out = root / "portfolio-stability"
    out.mkdir(exist_ok=True)
    recipes, seals = {}, {}
    for arm, path in plan["runs"].items():
        folder = root / path
        seal = read(folder / "sealed.json")
        final = read(folder / "run/agent/final_selection.json")
        require(seal["independent_reload"] and seal["model_sha256"] == final["model_sha256"]
                and seal["members"] == final["ensemble_components"], "Election identity changed")
        weights = final.get("weights")
        require(not weights or (len(weights) == len(seal["members"]) and np.isclose(sum(weights), 1)),
                "Inconsistent deployed weights")
        recipes[arm] = recipe(seal["members"], weights)
        seals[arm] = sha(folder / "sealed.json")
    members = sorted({m for r in recipes.values() for m in r["members"]})
    agent = root / plan["runs"][REVISED] / "run/agent"
    source = agent / "refit_data.joblib"
    data = joblib.load(source)
    metric = MetricSpec.from_payload(data)
    labels = np.asarray(data["y_train"])
    features = data["X_train"]
    require(metric.name == "accuracy" and data["pair_semantics"] == "preference"
            and len(labels) == read(root / plan["runs"][REVISED] / "sealed.json")["rows"],
            "Incompatible training payload")
    groups = sample_groups(TrainingTask(data["kind"], features, labels), "shared_text")
    conf = plan["assignments"]
    seeds = [int(hashlib.sha256(f"{conf['seed_domain']}/{i}".encode()).hexdigest()[:8], 16)
             for i in conf["seed_indices"]]
    old = read(root / "regrouped-runs/lock.json")
    require(not set(seeds) & set(old["seeds"]), "Repeated a previous assignment seed")
    prior = {read(p)["held_sha256"] for p in (root/"regrouped-runs").glob("seed_*/fold_*/partition.json")}
    prior.update(fingerprint(np.asarray(read(p)["held"])) for p in agent.glob("*_split.json")
                 if p.name in {"primary_split.json", "confirmation_1_split.json", "confirmation_2_split.json"})
    expected_fits = len(seeds) * conf["folds"] * len(members)
    require(expected_fits <= 30, "Supervised budget exceeded")
    lock(out/"inputs.json", dict(protocol=sha(PLAN), training=sha(source), seals=seals, recipes=recipes,
                                 seeds=seeds, rows=len(labels), groups=len(set(groups)), planned_fits=expected_fits))
    physical = out / "physical.jsonl"
    weights_dir = root / "discovery-library"
    settings = env(weights_dir, physical)
    activate_resource_directory(weights_dir, settings)
    settings.update({PREFERENCE_CATALOG_ENV: "all-installed", "SUPERAGENT_FROZEN_BATCH_REUSE": "1",
                     "SUPERAGENT_PAIRED_FEATURE_CACHE": str(root / "feature_bank")})
    records, totals = [], {}
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=conf["folds"], shuffle=True, random_state=seed)
        rights = {arm: np.zeros(len(labels), dtype=bool) for arm in recipes}
        visited = np.zeros(len(labels), dtype=bool)
        for fold, (fit, held) in enumerate(splitter.split(groups, labels, groups), 1):
            folder = out / f"seed_{seed}" / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            held_key = fingerprint(held)
            require(held_key not in prior and not np.any(visited[held])
                    and not set(groups[fit]) & set(groups[held]), "Repeated or overlapping test groups")
            prior.add(held_key)
            visited[held] = True
            payload = folder / "training.joblib"
            if not payload.exists():
                joblib.dump(dict(kind=data["kind"], X_train=take(features, fit), y_train=labels[fit],
                                 X_search=take(features, held), y_search=labels[held], metric_spec=metric.record(),
                                 pair_semantics=data["pair_semantics"], capture_probabilities=True, retain_model=False),
                            payload)
            lock(folder/"partition.json", dict(fit=fingerprint(fit), held=held_key, payload=sha(payload)))
            bank = {m: np.asarray(measure(m, payload, agent/"runtime_registry_0.json", folder/m,
                                         settings, metric, labels[held])) for m in members}
            for arm, selected in recipes.items():
                rights[arm][held] = correctness(bank, labels[held], np.asarray(metric.labels), selected)
            summary = dict(seed=seed, fold=fold, rows=len(held),
                           correct={k: int(v[held].sum()) for k, v in rights.items()},
                           paired={k: paired(rights[REVISED][held], rights[k][held]) for k in plan["comparisons"]})
            lock(folder/"metrics.json", summary)
            records.append(summary)
            print(seed, fold, summary["correct"], flush=True)
        require(visited.all(), "Incomplete out-of-fold coverage")
        totals[str(seed)] = dict(correct={k: int(v.sum()) for k, v in rights.items()},
                                 paired={k: dict(**paired(rights[REVISED], rights[k]),
                                                 cluster_interval=group_interval(rights[REVISED], rights[k], groups, seed))
                                         for k in plan["comparisons"]})
    positive = {k: sum(r["paired"][k]["net"] > 0 for r in records) for k in plan["comparisons"]}
    passed = all(positive[k] >= 7 and all(v["paired"][k]["net"]/len(labels) >= .003 for v in totals.values())
                 for k in plan["comparisons"])
    actual = len(physical.read_text().splitlines())
    require(actual == expected_fits, "Independent fit count differs from the protocol")
    record = dict(inputs=sha(out/"inputs.json"), rows_per_assignment=len(labels), totals=totals, folds=records,
                  positive_folds=positive, consistent_vs_both=passed, supervised_fits=actual, model_api_requests=0,
                  designated_validation_opened=False, official_test_opened=False, H2_opened=False)
    lock(out/"results.json", record)
    print(totals, positive, passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda s: Path(s).resolve(), required=True)
    run(parser.parse_args().root)
