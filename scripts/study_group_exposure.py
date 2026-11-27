"""Replay learned group-weight proposals from a real trajectory on disjoint training folds."""
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
from superagent.autonomous_run import planner_observation
from superagent.group_exposure import errors, proposals, sizes
from superagent.metrics import MetricSpec
from superagent.paired_encoder import NAMES

PLAN = ROOT / "research/paired_encoder_search_1127/group_feedback_protocol.json"


def lock(path, obj):
    if path.is_file():
        require(read(path) == obj, "The locked study inputs changed")
    else:
        save(path, obj)


def run(root, weights):
    out = root / "group-feedback"
    out.mkdir(exist_ok=True)
    agent = root / "resource-raw/resource_default/run/agent"
    history = read(agent / "history.json")
    primary = agent / "primary.joblib"
    data = joblib.load(primary)
    metric = MetricSpec.from_payload(data)
    labels = np.asarray(data["y_search"])
    group_sizes = sizes(data["X_search"])
    record = dict(sha256=sha(primary), labels=labels)
    for row in history:
        if row["status"] == "completed" and row["split_id"] == "primary" and row["candidate"] in NAMES:
            item = joblib.load(agent / f"trial_{row['trial']}.joblib")
            observation = planner_observation(item, item["metrics"], record, metric)
            row["connected_group_errors"] = errors(group_sizes, labels, observation["predictions"])
    additions = proposals(history, set(r["candidate"] for r in history))
    sources = {s["template"] for s in additions.values()}
    evidence = {r["candidate"]: r["connected_group_errors"] for r in history if r["candidate"] in sources}
    base_registry = read(agent / "runtime_registry_0.json")
    registry = out / "registry.json"
    lock(registry, dict(base_registry, entries={**base_registry["entries"], **additions}))
    primary_reference = read(root / "scale-screen/results.json")["candidate_locked"]
    control = recipe(primary_reference)
    lock(out / "proposal_lock.json", dict(protocol=sha(PLAN), registry=sha(registry), feedback=evidence,
                                         proposals=additions, reference=control))
    partitions = [set(read(agent / (fold+"_split.json"))["held"]) for fold in FOLDS]
    require(all(a.isdisjoint(b) for a, b in combinations(partitions, 2)), "Held-out rows overlap")
    configure_weights(weights)
    counts = out / "physical.jsonl"
    settings = env(weights, counts)
    settings.update({PREFERENCE_CATALOG_ENV: "all-installed", "SUPERAGENT_PAIRED_FEATURE_CACHE": str(root / "feature_bank")})
    inherited = available(root)

    def evaluate(fold, options):
        if not options:
            return []
        bank, y, classes, _ = matrices(agent, fold, primary_reference, inherited[fold])
        original = correctness(bank, y, classes, control)
        for member in sorted({name for _, rule in options for name in rule["members"] if name in additions}):
            bank[member] = np.asarray(measure(member, agent/(fold+".joblib"), registry, out/fold/member,
                                               settings, metric, y), dtype=float)
        observations = []
        for name, rule in options:
            after = correctness(bank, y, classes, rule)
            observations.append(dict(candidate=name, template=additions[name]["template"], power=additions[name]["power"],
                                     recipe=rule, fold=fold, rows=len(y), reference=int(sum(original)), correct=int(sum(after)),
                                     gained=int(sum(after & ~original)), lost=int(sum(original & ~after)),
                                     net=int(after.sum()-original.sum())))
        return observations

    options = [(n, rule) for n in sorted(additions) for rule in (recipe([n]), recipe([*primary_reference,n],[.25,.25,.5]))]
    measured = evaluate("primary", options)
    nominated = []
    for source in sorted(sources):
        eligible = [r for r in measured if r["template"] == source and r["net"] > 0]
        if eligible:
            nominated.append(min(eligible, key=lambda r: (-r["net"], len(r["recipe"]["members"]), r["candidate"])))
    lock(out / "primary_lock.json", dict(measured=measured, nominated=nominated))
    print("primary", [(r["template"], r["power"], len(r["recipe"]["members"]), r["net"]) for r in measured], flush=True)
    checks = evaluate("confirmation_1", [(r["candidate"], r["recipe"]) for r in nominated])
    eligible = []
    for a, b in zip(nominated, checks):
        pooled = (a["net"]+b["net"]) / (a["rows"]+b["rows"])
        if b["net"] > 0 and pooled >= .003:
            eligible.append((min(a["net"]/a["rows"], b["net"]/b["rows"]), pooled, b))
    selected = sorted(eligible, key=lambda t:(-t[0],-t[1],len(t[2]["recipe"]["members"]),t[2]["candidate"]))
    winner = selected[0][2] if selected else None
    lock(out/"confirmation_lock.json", dict(measured=checks, finalist=winner))
    third = evaluate("confirmation_2", [(winner["candidate"], winner["recipe"])]) if winner else []
    fits = len(counts.read_text().splitlines()) if counts.exists() else 0
    require(fits == len(additions) + len(nominated) + bool(winner) <= 7, "Unexpected physical fit count")
    result = dict(protocol_sha256=sha(PLAN), feedback=evidence, proposals=additions, primary=measured, confirmation=checks,
                  third=third, advance_to_raw=bool(third and third[0]["net"]>0), actual_fits=fits,
                  new_model_api_requests=0, released_validation_opened=False, H2_opened=False, formal_test_opened=False)
    lock(out/"results.json", result)
    print("confirm", [(r["template"],r["net"]) for r in checks], "third", third, "fits", fits, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "weights"):
        p.add_argument("--"+name, type=lambda s:Path(s).resolve(), required=True)
    args=p.parse_args()
    run(args.root,args.weights)
