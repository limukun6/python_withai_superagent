"""Optional feedback-based third member for a completed unified preference run."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import shutil

import joblib
import numpy as np
from scipy.special import expit

from scripts.unified_released_study import read, require, sha
from scripts.unified_third_member import counts, same_split
from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.preference_handoff import design, fit_policy, inputs, select
from unified_agent import save


@dataclass
class FeedbackPreferenceVote(PreferenceProbabilityVote):
    """A portable native accuracy predictor for a confirmed soft routing rule."""
    policy: dict

    def predict_proba(self, rows):
        if not len(rows):
            return np.empty((0, 2))
        scores = np.column_stack([canonical_probabilities(m, rows)[1][:, 1] for m in self.models])
        data = inputs(scores, rows)
        w = expit(design(data, self.policy) @ self.policy["coefficients"])
        p = np.where(data["conflict"], (1-w)*data["pair"] + w*scores[:, 2], data["pair"])
        return np.column_stack((1-p, p))


def held_feedback(agent, split, records, full_task, groups, excluded, members, observed, y):
    manifest = read(agent / (split + "_split.json"))
    ids = np.asarray(manifest["search_indices"])
    seen_ids = fingerprint(ids.tolist())
    expected = joblib.load(agent / (split + ".joblib"))["X_search"]
    require(seen_ids == manifest["row_ids_sha256"] and len(ids) == len(set(ids.tolist()))
            and np.array_equal(y, full_task.y_train[ids]) and [full_task.X_train[i] for i in ids] == list(expected)
            and all(fingerprint(row["row_ids"]) == seen_ids for row in records if row["split_id"] == split),
            "Native feedback has inconsistent row identity or order")
    require(set(groups[ids]).isdisjoint(groups[np.asarray(manifest["train_indices"])]),
            "A headline component appears in both base fitting and feedback")
    kept = np.asarray([g not in excluded for g in groups[ids]], dtype=bool)
    used = ids[kept]
    if not len(used):
        return None
    x = np.column_stack([observed[k][0][kept] for k in members])
    return (inputs(x, [full_task.X_train[i] for i in used]), y[kept], set(groups[used]),
            {"split": split, "rows": len(used), "excluded": int((~kept).sum()), "row_digest": fingerprint(used.tolist())})


def learn_from_run(agent, folder, selection, spec, result):
    members = selection.get("ensemble_components", [])
    schema = spec.get("input_schema", {})
    if (selection.get("family") != "equal_soft_vote" or len(members) != 2 or len(set(members)) != 2
            or spec.get("metric") != "accuracy" or schema.get("pair_semantics") != "preference"
            or schema.get("pair_label_one") != "left" or set(spec.get("metric_spec", {}).get("labels", [])) != {0, 1}):
        return "outside the confirmed binary preference contract"
    if not all((agent / name).is_file() for name in ("classifier_fits.json", "prediction_evidence.json", "refit_data.joblib")):
        return "native fit records or supervised feedback unavailable"
    ledger = read(agent / "classifier_fits.json")
    count = ledger["attempted_classifier_fits"]
    result["prior_native_fits"] = count
    if ledger["unknown_failed_attempts"] or count != ledger["completed_classifier_fits"] or not 0 <= count < 11:
        return "no confirmed fit remaining in the 11-fit path"

    records = read(agent / "prediction_evidence.json")
    splits = {r["split_id"] for r in records}
    if "primary" not in splits:
        return "no opening evidence"
    y, ties, opening = same_split(agent, "primary", records)
    if not set(members) <= opening.keys() or len(opening) < 3:
        return "no additional scored candidate"
    options = {name: counts(y, ties, opening, members, name) for name in opening if name not in members}
    third = max(options, key=lambda k: (options[k]["net"], options[k]["rescued_by_candidate_alone"], -options[k]["trial"]))
    names = [*members, third]
    refit = joblib.load(agent / "refit_data.joblib")
    task = SimpleNamespace(kind="pair_text", X_train=refit["X_train"], y_train=refit["y_train"])
    groups = sample_groups(task, "shared_text")
    first = held_feedback(agent, "primary", records, task, groups, set(), names, opening, y)
    result.update(members=members, challenger=third, opening_options=options)
    if first is None:
        return "no learning feedback"
    confirm = None
    for split in sorted(s for s in splits if s.startswith("confirmation_") and s[13:].isdigit()):
        labels, _, other = same_split(agent, split, records)
        if set(names) <= other.keys():
            confirm = held_feedback(agent, split, records, task, groups, first[2], names, other, labels)
            if confirm is not None:
                break
    if confirm is None:
        return "no disjoint confirmation with all three candidates"
    trial = int(opening[third][1])
    trial_path = agent / f"trial_{trial}.joblib"
    require(read(agent / f"trial_{trial}.fits.json")["candidate"] == third, "Challenger fit identity differs")
    fitted = joblib.load(trial_path)
    col = list(fitted["score_classes"]).index(1)
    require(np.array_equal(np.asarray(fitted["search_scores"])[:, col], opening[third][0]),
            "Challenger artifact differs from its opening evidence")
    result.update(learning=first[3], confirmation=confirm[3], challenger_trial=trial,
                  challenger_sha256=sha(trial_path))
    save(folder / "admission.json", {"prior_native_fits": count, "maximum_extra_fits": 1, "path_limit": 11})
    policy = fit_policy("soft", first[0], first[1])
    choice = select({"soft": policy}, first[0], first[1], confirm[0], confirm[1])
    result.update(decision=choice, policy=policy, chosen=choice["chosen"], extra_fits=policy["supervised_fits"])
    if choice["chosen"] == "baseline":
        return "confirmation retained the delivered pair"
    base = joblib.load(agent / selection["model_file"])["model"]
    require(type(base) is PreferenceProbabilityVote and len(base.models) == 2, "Incompatible native pair")
    model = FeedbackPreferenceVote("pair_text", [*base.models, fitted["model"]], np.asarray([0, 1]), policy)
    model_path = folder / "selected_model.joblib"
    with model_path.open("xb") as stream:
        joblib.dump({"kind": "pair_text", "candidate": "conditional_preference_vote", "model": model}, stream)
    shutil.copyfile(agent / "task_spec.json", folder / "task_spec.json")
    save(folder / "final_selection.json", {"selected_experiment": "confirmed_conditional_preference",
         "family": "conditional_preference_vote", "model_format": "candidate_bundle_v1",
         "model_file": model_path.name, "model_sha256": sha(model_path), "task_spec_sha256": result["spec_sha256"],
         "test_accessed": False, "components": names, "challenger_source": "retained opening trial"})
    result["optional_selection_sha256"] = sha(folder / "final_selection.json")
    return "confirmation admitted the conditional third member"


def frozen(run):
    agent, folder = run / "agent", run / "conditional_handoff"
    receipt = read(folder / "decision.json")
    require(sha(agent / "final_selection.json") == receipt["source_selection_sha256"]
            and sha(agent / "task_spec.json") == receipt["spec_sha256"], "Native selection or contract changed")
    if receipt["chosen"] == "soft":
        require(sha(folder / "final_selection.json") == receipt["optional_selection_sha256"],
                "Conditional selection changed")
        return receipt, folder
    require(receipt["chosen"] == "baseline", "Unknown handoff choice")
    return receipt, agent


def attach(run):
    run = Path(run)
    agent, folder = run / "agent", run / "conditional_handoff"
    if (folder / "decision.json").exists():
        return frozen(run)[0]
    selection, spec = read(agent / "final_selection.json"), read(agent / "task_spec.json")
    require(not selection["test_accessed"] and sha(agent / selection["model_file"]) == selection["model_sha256"]
            and sha(agent / "task_spec.json") == selection["task_spec_sha256"], "Native delivery changed")
    # An incomplete writer keeps its admission. It must never silently consume a second training fit.
    folder.mkdir(exist_ok=False)
    result = {"source_selection_sha256": sha(agent / "final_selection.json"), "model_sha256": selection["model_sha256"],
              "spec_sha256": selection["task_spec_sha256"], "chosen": "baseline", "extra_fits": 0, "model_api_calls": 0}
    result["reason"] = learn_from_run(agent, folder, selection, spec, result)
    pending = folder / "decision.pending.json"
    save(pending, result)
    pending.rename(folder / "decision.json")
    return result
