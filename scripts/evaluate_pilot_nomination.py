"""Freeze raw pilot and refit predictions from both autonomous arms, then score grouped outer labels."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_crossfit_challenger import native_predict, study_splits
from scripts.study_encoder_fusion import save_csv
from scripts.study_pilot_nomination import PLAN, PUBLIC_PRIOR, current_components, previous_components
from scripts.unified_released_study import read, require, save, sha
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from tools.challenger_training import score_members, ties
from tools.pilot_challenger import STRATEGIES
from tools.portfolio_training import configure, fit_one
from tools.preference_handoff import binary, changes


def project(names, matrix, rows):
    base = matrix[:, :2].mean(axis=1)
    tie = ties(rows)
    votes = {"baseline": binary(base, tie)}
    for i, name in enumerate(names[2:], 2):
        votes[name] = binary((2*base + matrix[:, i])/3, tie)
    return votes


def columns_for(pilots, refits, decisions, previous):
    output = {"baseline": refits["baseline"], "previous_agent": previous, "pilot_baseline": pilots["baseline"]}
    for name in pilots:
        if name != "baseline":
            output["pilot_" + name] = pilots[name]
    for strategy, decision in decisions.items():
        for key in ("approved", "nominee"):
            member = decision[key] or "baseline"
            output[f"{strategy}_{key}"] = refits[member]
            output[f"pilot_{strategy}_{key}"] = pilots[member]
    return output


def store_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items()),
                    "Previously recorded outer probabilities differ")
    else:
        np.savez_compressed(path, **arrays)


def freeze_fold(folder, i, binding, prior, weights, cache):
    arms = {s: folder / s for s in STRATEGIES}
    task = load_training_task(folder / "train.zip")
    package = sha(folder / "train.zip")
    require(package == binding["train_packet"] and sha(folder / "features.zip") == binding["features_packet"],
            "Input differs from the registered packets")
    choices, completions, native, records = {}, {}, {}, {}
    for s, arm in arms.items():
        run = arm / "run"
        end = read(run / "pilot_completion.json")
        require(end["model_sha256"] == sha(run / "agent/selected_model.joblib") and
                end["decision_sha256"] == sha(run / "decision.json"), "Decision or model not frozen")
        choices[s], completions[s] = read(run / "decision.json"), end
        native[s] = native_predict(arm, weights)
        for name in ("run/pilot_protocol.json", "run/pilot_completion.json", "run/nomination.json",
                     "run/decision.json", "run/feedback.npz", "run/agent/final_selection.json", "direct/predictions.csv"):
            records[f"{s}/{name}"] = sha(arm / name)
    first = arms[STRATEGIES[0]] / "run"
    names = read(first / "pilot_protocol.json")["inventory"]["candidates"]
    rows = load_prediction_features(folder / "features.zip", "pair_text", read(first / "agent/task_spec.json"))
    require(len(rows) == binding["outer"]["rows"], "Unexpected outer row count")
    used = set(d["nominee"] for d in choices.values())
    refit_names = [*names[:2], *[name for name in names[2:] if name in used]]
    pool = {}
    for source in [p for arm in arms.values() for p in current_components(arm)] + previous_components(prior, i):
        receipt = read(source / "trained.json")
        pool.setdefault((receipt["identity"]["ids"], receipt["name"]), source)
    full = np.arange(len(task.y_train))
    identity = dict(package=package, ids=fingerprint(full.tolist()))
    sources, added = {}, 0
    configure(weights, cache)
    with reuse_frozen_vectors():
        small_models = [joblib.load(first / "pilot" / n / "model.joblib") for n in names]
        small = score_members(small_models, rows)
        models = []
        for n in refit_names:
            path = pool.get((identity["ids"], n))
            if path is None:
                require(n not in names[:2], "Missing incumbent refit")
                path = folder / "shadow" / n
                fit_one(path, n, task.X_train, task.y_train, identity)
                added += 1
            rec = read(path / "trained.json")
            require(rec["identity"] == identity and rec["model"] == sha(path / "model.joblib") and
                    rec["ledger"] == sha(path / "fits.json"), "Mismatched comparison refit")
            sources[n] = {"model": rec["model"], "ledger": rec["ledger"], "identity": identity}
            models.append(joblib.load(path / "model.joblib"))
        final = score_members(models, rows)
    previous_receipt = read(PUBLIC_PRIOR / folder.name / "sealed.json")["records"]["direct/predictions.csv"]
    previous_path = PUBLIC_PRIOR / folder.name / "direct/predictions.csv"
    require(sha(previous_path) == previous_receipt, "Different historical agent export")
    previous = load_export(previous_path)
    require(np.array_equal(previous["row_id"], np.arange(len(rows))), "Previous row order differs")
    columns = columns_for(project(names, small, rows), project(refit_names, final, rows), choices, previous["prediction"])
    require(all(np.array_equal(native[s], columns[s + "_approved"]) for s in STRATEGIES),
            "Cold raw-text delivery differs from the frozen model probabilities")
    store_arrays(folder / "outer.npz", {"pilot": small, "refit": final})
    row_path = folder / "rows.csv"
    if row_path.exists():
        saved = load_export(row_path)
        require(np.array_equal(saved.pop("row_id"), np.arange(len(rows))) and saved.keys() == columns.keys() and
                all(np.array_equal(saved[k], v) for k, v in columns.items()), "Existing outer rows differ")
    else:
        save_csv(row_path, range(len(rows)), columns)
    records.update({f: sha(folder / f) for f in ("outer.npz", "rows.csv")})
    actual = sum(c["new_classifier_fits"] for c in completions.values()) + added
    require(actual <= (10 if i == 0 else 5) and len(names) + len(refit_names) <= 10, "Paired fit budget exceeded")
    summary = dict(records=records, pilot_members=names, refit_members=refit_names, refit_records=sources,
                   prior_predictions=previous_receipt, shadow_new=added, total_new=actual,
                   shadow_names=sorted(p.name for p in (folder / "shadow").glob("*") if p.is_dir()),
                   native_seconds={s: read(arm / "direct/elapsed.json")["seconds"] for s, arm in arms.items()})
    save(folder / "sealed.json", summary)
    print(folder.name, "native rows", len(rows), "new including controls", actual, flush=True)


def freeze(out, prior, weights, cache):
    spec = read(out / "protocol.json")
    require(spec["plan"] == sha(PLAN), "Plan changed")
    seals = []
    for i in range(5):
        folder = out / f"fold_{i}"
        if not (folder / "sealed.json").exists():
            freeze_fold(folder, i, spec["folds"][str(i)], prior, weights, cache)
        require(all(sha(folder / k) == v for k, v in read(folder / "sealed.json")["records"].items()), "Fold changed")
        seals.append(sha(folder / "sealed.json"))
    save(out / "frozen.json", dict(protocol=sha(out / "protocol.json"), folds=seals))


def score(out, archive):
    bound = read(out / "frozen.json")
    require(bound["protocol"] == sha(out / "protocol.json"), "Study protocol changed")
    task, groups, splits = study_splits(archive)
    folds, all_columns, indices = [], {}, []
    seen_groups = set()
    for i, (_, held) in enumerate(splits):
        folder = out / f"fold_{i}"
        seal = read(folder / "sealed.json")
        require(sha(folder / "sealed.json") == bound["folds"][i] and
                all(sha(folder / k) == v for k, v in seal["records"].items()), "Unsealed score")
        bind = read(out / "protocol.json")["folds"][str(i)]["outer"]
        require(fingerprint(held.tolist()) == bind["ids"] and not seen_groups.intersection(groups[held]), "Group overlap")
        seen_groups.update(groups[held])
        ids = held.tolist()
        pred = load_export(folder / "rows.csv")
        require(np.array_equal(pred.pop("row_id"), np.arange(len(held))), "Outer order mismatch")
        y = task.y_train[held]
        metrics = {k: changes(pred["pilot_baseline" if k.startswith("pilot_") else "baseline"], v, y)
                   for k, v in pred.items()}
        choices = {}
        for s in STRATEGIES:
            end = read(folder / s / "run/pilot_completion.json")
            d = read(folder / s / "run/decision.json")
            choices[s] = dict(nominee=d["nominee"], approved=d["approved"], confirm=d["confirm"], threshold=d["minimum_net"],
                              nomination_net=(read(folder / s / "run/nomination.json")["learning"][d["nominee"]]["net"]
                                              if d["nominee"] else None), cold_fits=end["cold_classifier_fits"],
                              new_fits=end["new_classifier_fits"], reused_fits=end["reused_classifier_fits"],
                              seconds=end["elapsed_seconds"], native_seconds=seal["native_seconds"][s])
        folds.append(dict(fold=i, assessments=metrics, choices=choices, rows=len(held), new_fits=seal["total_new"],
                          shadow_fits=seal["shadow_new"]))
        indices.extend(ids)
        for k, v in pred.items():
            all_columns.setdefault(k, []).append(v)
    ids = np.asarray(indices)
    require(len(ids) == len(set(indices)) == 6608 and len(seen_groups) == 4874, "Coverage differs")
    y, g = task.y_train[ids], groups[ids]
    p = {k: np.concatenate(v) for k, v in all_columns.items()}
    arms = {k: comparison(p["pilot_baseline" if k.startswith("pilot_") else "baseline"], v, y, g)
            for k, v in p.items()}
    arms["direct_versus_previous"] = comparison(p["previous_agent"], p["pilot-accuracy_approved"], y, g)
    arms["brier_versus_direct"] = comparison(p["pilot-accuracy_approved"], p["pilot-brier_approved"], y, g)
    wins = {s: sum(f["assessments"][s+"_approved"]["net"] > 0 for f in folds) for s in STRATEGIES}
    primary = arms[STRATEGIES[0]+"_approved"]
    passes = wins[STRATEGIES[0]] >= 4 and primary["net"] >= 5 and primary["paired_95_fraction"][0] > 0
    new = sum(f["new_fits"] for f in folds)
    require(new <= 30, "Study-wide research fit budget exceeded")
    result = dict(frozen=sha(out / "frozen.json"), rows=len(y), groups=len(seen_groups), folds=folds, arms=arms,
                  positive_folds=wins, passes=passes, actual_new_classifier_fits=new,
                  model_api_calls=0, model_api_rmb=0, designated_validation_used=False, formal_test_used=False)
    save(out / "result.json", result)
    print({s: (arms[s+"_approved"]["correct"], arms[s+"_approved"]["net"], wins[s]) for s in STRATEGIES},
          "primary passes", passes, "fits", new)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "score"))
    p.add_argument("--out", required=True, type=lambda s: Path(s).resolve())
    for arg in ("prior", "weights", "cache", "archive"):
        p.add_argument("--"+arg, type=lambda s: Path(s).resolve())
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.out, a.prior, a.weights, a.cache)
    else:
        score(a.out, a.archive)
