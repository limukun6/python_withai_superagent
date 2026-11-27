"""Use a held-out training pilot to decide whether to average independent fits of the selected members."""
import time
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .ab_run import sha
from .experiment_policy import identity
from .models import SoftVoteModel, predict_with_model, probability_with_model
from .score_evidence import checked_score_record
from .splits import sample_groups, take
from .task_semantics import declared_semantics
from .training_randomness import fitted_seed

SEEDS = (13291, 94723)


def supports_repeat(model, capability):
    if capability["classifier_fits"] != 1 or capability["refit_classifier_fits"] != 1:
        return False
    if isinstance(model, Pipeline):
        return (capability["params"].get("runtime_composed") is True
                and "random_state" in model.steps[-1][1].get_params(deep=False))
    return fitted_seed(model) is not None


def evidence(metric, labels, original, adjusted, groups):
    classes = np.asarray(metric.labels)
    truth = np.asarray(labels)
    target = (truth[:, None] == classes).astype(float)
    old, new = (p / p.sum(1, keepdims=True) for p in (original, adjusted))
    old_hard, new_hard = classes[old.argmax(1)], classes[new.argmax(1)]
    error_change = ((new - target) ** 2 - (old - target) ** 2).sum(1)
    def difference(mask):
        prior = metric.evaluate(truth[mask], old_hard[mask], old[mask], classes)["score"]
        later = metric.evaluate(truth[mask], new_hard[mask], new[mask], classes)["score"]
        return float(later - prior), float(error_change[mask].mean())
    _, inverse = np.unique(groups, return_inverse=True)
    count = min(12, len(np.unique(inverse)))
    if count < 4:
        return dict(accept=False, reason="Too few independent groups for a paired estimate")
    blocks = inverse % count
    all_rows = np.ones(len(truth), dtype=bool)
    metric_delta, brier_delta = difference(all_rows)
    changes = np.asarray([difference(blocks != i) for i in range(count)])
    errors = np.sqrt((count - 1) / count * np.square(changes - changes.mean(0)).sum(0))
    diversity = float(np.sqrt(np.mean(np.square(new - old))))
    accepted = (diversity >= .002 and metric_delta >= -max(1 / len(truth), 1.64 * errors[0])
                and brier_delta <= max(1e-8, 1.64 * errors[1]))
    return dict(accept=bool(accepted), rows=len(truth), group_blocks=count, score_gain=metric_delta,
                brier_increase=brier_delta, score_se=float(errors[0]), brier_se=float(errors[1]),
                rms_probability_change=diversity, labels_changed=int(np.sum(old_hard != new_hard)))


def after_refit(task, spec, metric, choice, selection, observed, history, payloads, executor, directory, log, timeout):
    members = choice["members"]
    base_identity = selection["model_sha256"]
    report = dict(members=members, incumbent=base_identity, seeds=list(SEEDS), stage="eligibility")
    def retain(reason):
        log.write_json("seed_replication.json", dict(report, action="retain", reason=reason))
        return selection

    if "linear_only" in spec.get("method_restrictions", ()) or declared_semantics(task) == "preference":
        return retain("Keep the selected model's required decision form")
    if (task.spec.get("evaluation_protocol") or {}).get("kind", "grouped_stratified") != "grouped_stratified":
        return retain("The repeat pilot requires disjoint grouped training feedback")
    matching = [name for name, bank in observed.items() if all(m in bank and bank[m].get("probabilities") is not None for m in members)]
    if len(matching) < 2:
        return retain("The selected solution needs probabilities on two training partitions")
    split = matching[-1]
    records = payloads[split]
    if len(records["labels"]) < 128:
        return retain("The training pilot has fewer than 128 observations")
    try:
        if sha(directory / selection["model_file"]) != base_identity:
            raise ValueError("Original delivery changed before the repeat pilot")
        base_models = [joblib.load(directory / f"refit_component_{i}.joblib")["model"] for i in range(len(members))]
        eligible = [i for i,m in enumerate(members) if supports_repeat(base_models[i], executor.caps[m])]
        if not eligible:
            return retain("Selected members have no supported single-fit randomization")
        left = executor.limit - executor.accounting()["attempted_classifier_fits"]
        need = 3 * len(eligible)
        report.update(eligible=[members[i] for i in eligible], pilot_split=split, required_extra_fits=need, fits_available=left)
        if left < need or executor.accounting()["unknown_failed_attempts"]:
            return retain("Insufficient accounted fits for a pilot and both full repeats")
        times = [r.get("training_seconds", 0.) for r in history if r["status"] == "completed" and r["candidate"] in report["eligible"]]
        if executor.wall - (time.monotonic() - executor.start) < need * max([2., *times]):
            return retain("Insufficient time for the measured fit cost")
        if sha(records["path"]) != records["sha256"]:
            raise ValueError("Pilot partition changed")
        pilot = joblib.load(records["path"])
        classes = np.asarray(metric.labels)
        original = [observed[split][name]["probabilities"] for name in members]
        old = np.mean(original, axis=0)
        if len(members) == 1 and not np.array_equal(observed[split][members[0]]["predictions"], classes[old.argmax(1)]):
            return retain("The selected classifier has a distinct hard-label decision rule")
        report["stage"] = "pilot"
        pilot_path = directory / "seed_pilot_data.joblib"
        joblib.dump(dict(pilot, supervised_random_seed=SEEDS[0]), pilot_path)
        pilot_hash = sha(pilot_path)
        averaged = list(original)
        for i in eligible:
            name = members[i]
            destination = directory / f"seed_pilot_{i}.joblib"
            _, facts = executor.worker(pilot_path, name, destination, timeout)
            if facts["resolved_configuration"].get("training_random_seed") != SEEDS[0]:
                raise ValueError("Unverified pilot randomization")
            record = joblib.load(destination)["prediction_record"]
            if record["evaluation_payload_sha256"] != pilot_hash:
                raise ValueError("Pilot feedback belongs to a different partition")
            p, order = checked_score_record(record["score_record"], record, metric, name, records["labels"])
            averaged[i] = (original[i] + p[:, [list(order).index(c) for c in classes]]) / 2
        groups = sample_groups(SimpleNamespace(kind=task.kind, X_train=pilot["X_search"]), "shared_text")
        pilot_result = evidence(metric, records["labels"], old, np.mean(averaged, axis=0), groups)
        report.update(pilot=pilot_result, pilot_payload=pilot_hash)
        if not pilot_result["accept"]:
            return retain("The repeated fit offered too little diversity or worsened the training feedback")
        report["stage"] = "full_refits"
        log.write_json("seed_replication.json", dict(report, action="refit"))
        full = joblib.load(directory / "refit_data.joblib")
        parts = [[model] for model in base_models]
        for seed in SEEDS:
            path = directory / f"seed_refit_data_{seed}.joblib"
            joblib.dump(dict(full, supervised_random_seed=seed), path)
            for i in eligible:
                destination = directory / f"seed_refit_{i}_{seed}.joblib"
                model, facts = executor.worker(path, members[i], destination, timeout)
                if facts["resolved_configuration"].get("training_random_seed") != seed:
                    raise ValueError("Unverified full-data randomization")
                parts[i].append(model)
        def vote(models):
            return models[0] if len(models) == 1 else SoftVoteModel(task.kind, models, classes)
        model = vote([vote(values) for values in parts])
        family = "seed_averaged_vote"
        destination = directory / "seed_averaged_model.joblib"
        report["stage"] = "export"
        joblib.dump(dict(kind=task.kind, candidate=family, model=model), destination)
        probe = take(task.X_train, np.linspace(0, len(task.y_train)-1, min(12, len(task.y_train)), dtype=int))
        restored = joblib.load(destination)["model"]
        np.testing.assert_array_equal(predict_with_model(task.kind, model, probe), predict_with_model(task.kind, restored, probe))
        a, b = (probability_with_model(task.kind, m, probe) for m in (model, restored))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], rtol=1e-7, atol=1e-8)
        selected = dict(selection, selected_experiment="seed_" + identity([members, SEEDS])[:12],
                        family=family, model_file=destination.name, model_sha256=sha(destination),
                        seed_replication=dict(seeds=list(SEEDS), members_repeated=report["eligible"], pilot_split=split,
                                              prior_model_sha256=base_identity, base_member_weights="equal",
                                              prior_selection_score=selection["selection_score"]),
                        selection_metric="ordinary training folds and paired repeat pilot")
        log.write_json("seed_replication.json", dict(report, action="adopt", model=selected["model_sha256"]))
        log.write_json("final_selection.json", selected)
        return selected
    except Exception as exc:
        return retain(f"Optional repeat failed during {report['stage']}: {type(exc).__name__}: {exc}")
