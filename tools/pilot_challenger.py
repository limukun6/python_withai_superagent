"""Nominate one challenger from matched pilot fits, then ask a separate set of groups to confirm it."""
import math
import time

import joblib
import numpy as np

from scripts.unified_encoder_portfolio import candidate_inventory
from scripts.unified_released_study import require, sha
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import rows_at, score_members, ties
from tools.crossfit_challenger import ranked_groups, role_indices
from tools.feedback_policy_pool import grouped_error
from tools.pilot_fit_cache import FitCache
from tools.portfolio_training import configure, fit_one, write_selection
from tools.preference_handoff import binary, changes
from unified_agent import save, train as common_train

STRATEGIES = ("pilot-accuracy", "pilot-brier")
BACKUP_STRATEGIES = ("pilot-no-backup", "pilot-backup")


def pilot_roles(groups, labels):
    previous = role_indices(groups)
    remaining = np.sort(np.concatenate((previous["b"], previous["nominate"])))
    ordered = ranked_groups(groups[remaining], "pilot-feedback-v1/")
    learn = np.isin(groups[remaining], ordered[::2])
    roles = dict(pilot=previous["a"], learn=remaining[learn], confirm=remaining[~learn])
    if any(len(set(groups[ix])) < 4 or set(labels[ix]) != {0, 1} for ix in roles.values()):
        return None
    return roles


def compare_vote(base, third, tie, labels, groups):
    combined = (2 * base + third) / 3
    gain = (base - labels) ** 2 - (combined - labels) ** 2
    observed = changes(binary(base, tie), binary(combined, tie), labels)
    mean, se = float(gain.mean()), grouped_error(gain, groups)
    return {**observed, "brier_gain": mean, "brier_se": se, "brier_lower": mean - se}


def panel_scores(names, matrix, rows, labels, groups):
    require(matrix.shape == (len(labels), len(names)) and np.isfinite(matrix).all()
            and np.all((0 <= matrix) & (matrix <= 1)) and len(names) == len(set(names))
            and len(rows) == len(groups) == len(labels) and set(labels) == {0, 1}, "Unaligned pilot feedback")
    base = matrix[:, :2].mean(axis=1)
    tie = ties(rows)
    return {name: compare_vote(base, matrix[:, i], tie, labels, groups) for i, name in enumerate(names[2:], 2)}


def nominate(results, strategy):
    require(strategy in STRATEGIES, "Unknown pilot nomination strategy")
    first, second = ("net", "brier_gain") if strategy == "pilot-accuracy" else ("brier_lower", "net")
    order = sorted(results, key=lambda n: (-results[n][first], -results[n][second], n))
    return order[0] if order and results[order[0]][first] > 0 else None


def approve(name, learning, check, n):
    minimum = max(2, math.ceil(.002 * n))
    passed = bool(name and learning[name]["net"] > 0 and check["net"] >= minimum and check["brier_gain"] > 0)
    return dict(nominee=name, approved=name if passed else None, confirm=check, minimum_net=minimum)


def backup_roles(roles, groups, labels):
    if roles is None:
        return None
    remaining = roles["confirm"]
    ranked = ranked_groups(groups[remaining], "pilot-backup-check-v1/")
    first = np.isin(groups[remaining], ranked[::2])
    portions = {"first_check": remaining[first], "confirm": remaining[~first]}
    if any(len(set(groups[ix])) < 4 or set(labels[ix]) != {0, 1} for ix in portions.values()):
        return None
    return {**roles, **portions}


def shortlist(results):
    first = nominate(results, "pilot-accuracy")
    eligible = {n: row for n, row in results.items() if n != first and row["net"] > 0}
    return first, nominate(eligible, "pilot-brier")


def advance(first, backup, check, strategy):
    require(strategy in BACKUP_STRATEGIES, "Invalid backup strategy")
    if first is not None and check["net"] > 0 and check["brier_gain"] > 0:
        return first
    return backup if strategy == "pilot-backup" else None


def confirmation_for(candidate, ids, names, models, task, groups):
    if candidate is None:
        return np.empty((0, 0)), np.array([], dtype=int), None
    rows = rows_at(task, ids)
    p = score_members([*models[:2], models[names.index(candidate)]], rows)
    summary = compare_vote(p[:, :2].mean(1), p[:, 2], ties(rows), task.y_train[ids], groups[ids])
    return p, ids, summary


def choose_from_pilots(directory, strategy, task, names, models, groups, roles):
    learn = roles["learn"]
    rows = rows_at(task, learn)
    scores = score_members(models, rows)
    feedback = panel_scores(names, scores, rows, task.y_train[learn], groups[learn])
    arrays = dict(learn=scores, learn_ids=learn)
    extra = {}
    is_backup_study = strategy in BACKUP_STRATEGIES
    if is_backup_study:
        selected, backup = shortlist(feedback)
    else:
        selected = nominate(feedback, strategy)
    admission = dict(strategy=strategy, chosen=selected, learning=feedback, members=names,
                     learning_rows=fingerprint(learn.tolist()))
    if is_backup_study:
        admission["backup"] = backup
    save(directory / "nomination.json", admission)
    if is_backup_study:
        first_scores, ids, check = confirmation_for(selected, roles["first_check"], names, models, task, groups)
        arrays.update(first_check=first_scores, first_check_ids=ids)
        finalist = advance(selected, backup, check, strategy)
        stage = dict(first=selected, backup=backup, first_check=check, advanced=finalist,
                     nomination=sha(directory / "nomination.json"))
        save(directory / "first_check.json", stage)
        extra["first_check_sha256"] = sha(directory / "first_check.json")
        selected = finalist
    confirm = roles["confirm"]
    confirmed, ids, check = confirmation_for(selected, confirm, names, models, task, groups)
    decision = approve(selected, feedback, check, len(confirm))
    np.savez_compressed(directory / "feedback.npz", **arrays, confirm=confirmed, confirm_ids=ids)
    decision.update(nomination=sha(directory / "nomination.json"), feedback=sha(directory / "feedback.npz"), **extra)
    save(directory / "decision.json", decision)
    return decision


def train(package, directory, weights, feature_bank=None, cache_path=None, strategy=STRATEGIES[0]):
    require(strategy in (*STRATEGIES, *BACKUP_STRATEGIES), "Invalid pilot strategy")
    task = load_training_task(package)
    inventory = dict(candidates=[], excluded={}, reason="weight_directory_not_configured")
    if weights is not None:
        configure(weights, feature_bank)
        inventory = candidate_inventory(task, weights, l6_as_challenger=True)
    names = inventory["candidates"]
    roles = None
    if names:
        groups = sample_groups(task, "shared_text")
        roles = pilot_roles(groups, task.y_train)
        if strategy in BACKUP_STRATEGIES:
            roles = backup_roles(roles, groups, task.y_train)
        if roles is None:
            inventory["reason"] = "insufficient_disjoint_training_groups"
    if roles is None:
        status = common_train(package, directory)
        save(directory / "pilot_availability.json", inventory)
        return status

    require(len(names) <= 8, "Pilot and three reserved refits exceed the 11-fit path ceiling")
    package_id = sha(package)
    cache = FitCache(cache_path, package_id)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    roles_record = {k: dict(rows=len(v), groups=len(set(groups[v])), ids=fingerprint(v.tolist())) for k, v in roles.items()}
    save(directory / "pilot_protocol.json", dict(strategy=strategy, inventory=inventory, package=package_id,
         rows=len(task.y_train), roles=roles_record, cold_fit_ceiling=len(names) + 3, model_api_limit=0))
    actions = []

    def fit(name, stage, ids):
        path = directory / stage / name
        identity = {"package": package_id, "ids": fingerprint(ids.tolist())}
        reused = cache.restore(path, name, identity)
        record = fit_one(path, name, rows_at(task, ids), task.y_train[ids], identity)
        actions.append({"stage": stage, "candidate": name, "reused": reused,
                        "identity": identity, "model": record["model"], "ledger": record["ledger"]})
        return joblib.load(path / "model.joblib")

    with reuse_frozen_vectors():
        pilots = [fit(n, "pilot", roles["pilot"]) for n in names]
        decision = choose_from_pilots(directory, strategy, task, names, pilots, groups, roles)
        members = names[:2] + ([decision["approved"]] if decision["approved"] else [])
        full = np.arange(len(task.y_train))
        final = [fit(n, "refit", full) for n in members]
        vote = PreferenceProbabilityVote("pair_text", final, np.array([0, 1]))
        write_selection(directory / "agent", vote, members, task.spec, decision["approved"] or "baseline", strategy)
    fits = sum(not a["reused"] for a in actions)
    save(directory / "pilot_completion.json", dict(strategy=strategy, cold_classifier_fits=len(actions),
         new_classifier_fits=fits, reused_classifier_fits=len(actions)-fits, actions=actions,
         supervised_nominations=1, decision_sha256=sha(directory / "decision.json"),
         model_sha256=sha(directory / "agent/selected_model.joblib"), model_api_calls=0,
         elapsed_seconds=round(time.monotonic()-started, 4)))
    print(strategy, "nominee", decision["nominee"], "approved", decision["approved"], "fits", fits,
          "reused", len(actions)-fits, flush=True)
    return 0
