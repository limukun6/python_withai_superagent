"""Opt-in autonomous preference portfolio. The default unified entry remains available for other tasks."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np

from scripts.unified_released_study import read, require, sha
from superagent.ab_complement import canonical_probabilities
from superagent.ab_sentence import ALTERNATIVES, verified_directory, verified_files
from superagent.data import load_training_task
from superagent.predict import create_predictions
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.encoder_assets import MODELS, verified
from tools.portfolio_training import REFERENCE, configure, export, fit_one
from tools.preference_portfolio import choose, panel
from unified_agent import audit_partition, pair_packet, save, train as unified_train


def training_roles(groups, labels, allowed):
    known = sorted(set(groups[allowed]), key=lambda g: hashlib.sha256(("portfolio-feedback-v1/" + g).encode()).digest())
    if len(known) < 10:
        return None
    bucket = {g: i % 10 for i, g in enumerate(known)}
    bands = np.array([bucket[g] for g in groups[allowed]])
    parts = {"base": allowed[bands < 6], "learn": allowed[(bands >= 6) & (bands < 8)], "confirm": allowed[bands >= 8]}
    if any(set(labels[ix]) != {0, 1} for ix in parts.values()):
        return None
    return parts


def candidate_inventory(task, weights, *, l6_as_challenger=False):
    spec = task.spec
    schema = spec.get("input_schema", {})
    if (task.kind != "pair_text" or schema.get("pair_semantics") != "preference" or schema.get("pair_label_one") != "left"
            or spec.get("metric", "accuracy") != "accuracy" or set(task.y_train) != {0, 1}):
        return {"candidates": [], "reason": "incompatible_task_contract", "excluded": {}}
    e5 = ALTERNATIVES["e5"]
    required = ((REFERENCE[0], lambda: verified_directory(deep=True)),
                (REFERENCE[1], lambda: verified_files(e5["env"], e5["files"])),
                (REFERENCE[2], verified_directory))
    excluded = {}
    reference = REFERENCE[:2] if l6_as_challenger else REFERENCE
    for name, check in required[:len(reference)]:
        try:
            if not check():
                excluded[name] = "reference_missing_or_unverified"
        except OSError:
            excluded[name] = "unreadable_reference"
    if excluded:
        return {"candidates": [], "reason": "reference_unavailable", "excluded": excluded}
    extra = []
    if l6_as_challenger:
        try:
            if verified_directory():
                extra.append(REFERENCE[2])
            else:
                excluded[REFERENCE[2]] = "not_installed_or_unverified"
        except OSError:
            excluded[REFERENCE[2]] = "unreadable_asset"
    for name in MODELS:
        if not (weights/name).exists():
            excluded[name] = "not_installed"
            continue
        try:
            verified(weights, name)
        except ValueError:
            excluded[name] = "missing_or_mismatched_pinned_file"
            continue
        except OSError:
            excluded[name] = "unreadable_asset"
            continue
        extra.append(name)
    return {"candidates": [*reference, *extra] if extra else [], "excluded": excluded,
            "reason": "eligible" if extra else "no_verified_extra_encoder"}


def candidates(task, weights):
    return candidate_inventory(task, weights)["candidates"]


def train(package, directory, weights, cache):
    configure(weights, cache)
    task = load_training_task(package)
    inventory = candidate_inventory(task, weights)
    names = inventory["candidates"]
    roles = None
    if names:
        audit, reason = audit_partition(task)
        allowed = np.arange(len(task.y_train)) if audit is None else np.asarray(audit[0])
        groups = sample_groups(task, "shared_text")
        roles = training_roles(groups, task.y_train, allowed)
        if roles is None:
            inventory["reason"] = "insufficient_disjoint_training_groups"
    if roles is None:
        status = unified_train(package, directory)
        save(directory/"portfolio_availability.json", inventory)
        return status
    directory.mkdir(parents=True, exist_ok=False)
    reserve = np.array([], dtype=int) if audit is None else audit[1]
    memberships = {k: {"rows": len(v), "groups": len(set(groups[v])), "ids": fingerprint(v.tolist())}
                   for k, v in {**roles, "audit": reserve}.items()}
    seen = [set(groups[ix]) for ix in (*roles.values(), reserve)]
    require(sum(map(len, seen)) == len(set.union(*seen)), "Training and audit share a connected title")
    record = {"train_sha256": sha(package), "roles": memberships, "audit_reason": reason,
              "candidates": names, "availability": inventory, "maximum_classifier_fits": len(names), "maximum_supervised_routing_steps": 4,
              "path_limit": 11, "model_api_limit": 0}
    save(directory/"protocol.json", record)
    base = roles["base"]
    rows = [task.X_train[i] for i in base]
    fitted, models = [], []
    for name in names:
        folder = directory/"members"/name
        fitted.append(fit_one(folder, name, rows, task.y_train[base],
                              {"package": record["train_sha256"], "base": memberships["base"]["ids"]}))
        models.append(joblib.load(folder/"model.joblib"))
        print("fitted", name, flush=True)
    inputs = {}
    for role in ("learn", "confirm"):
        pairs = [task.X_train[i] for i in roles[role]]
        scores = np.column_stack([canonical_probabilities(m, pairs)[1][:, 1] for m in models])
        np.save(directory/(role+"_scores.npy"), scores, allow_pickle=False)
        inputs[role] = panel(scores, pairs)
    save(directory/"routing_admission.json", {"maximum_continuous": 2, "maximum_discrete": 2, "classifier_fits": len(names)})
    decision = choose(names, inputs["learn"], task.y_train[roles["learn"]], inputs["confirm"], task.y_train[roles["confirm"]])
    save(directory/"decision.json", decision)
    export(directory/"agent", models, decision, task.spec)
    if len(reserve):
        pair_packet(directory/"audit_features.zip", task, reserve, training=False)
        export(directory/"reference/agent", models, dict(decision, chosen="baseline"), task.spec)
    save(directory/"completion.json", {"members": fitted, "selected": decision["chosen"],
         "model_sha256": read(directory/"agent/final_selection.json")["model_sha256"],
         "classifier_fits": len(fitted), "continuous_fits": decision["continuous_fits"],
         "discrete_selections": decision["supervised_shortlists"], "model_api_calls": 0,
         "course_validation_used": False, "formal_test_used": False})
    print("frozen", decision["chosen"], "model", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("train", "predict"):
        cmd = sub.add_parser(name)
        cmd.add_argument("package", type=Path)
        cmd.add_argument("--run-dir", type=Path, required=True)
        cmd.add_argument("--weights", type=Path, required=True)
        cmd.add_argument("--cache", type=Path)
        if name == "predict":
            cmd.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "train":
        raise SystemExit(train(a.package, a.run_dir, a.weights.resolve(), a.cache))
    configure(a.weights, a.cache)
    print(create_predictions(str(a.package), a.run_dir/"agent", a.output))
