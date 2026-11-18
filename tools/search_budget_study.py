"""Compare 4-search/3-confirm and 6-search/2-confirm with a replayed 4/2 arm."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one, train_candidate
from superagent.confirmation import decide, shortlist
from superagent.data import DevelopmentTask, TrainingTask, load_training_task
from superagent.evidence_agent import route_of
from superagent.evidence_schedule import duel
from superagent.models import predict_with_model
from superagent.splits import effective_grouping, grouped_partition, overlap_audit, take
from tools.expanded_feedback_pilot import RESERVE_SEED, RUN_OPTIONS, SEARCH_SEED, frozen_control, prepare_task, protocol


SEEDS = (42011, 42017)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outer_partition(archive, seed):
    whole = load_training_task(archive)
    grouping = effective_grouping(whole.kind, "shared_text")
    keep, _, _ = grouped_partition(whole, RESERVE_SEED, grouping)
    dev = TrainingTask(whole.kind, take(whole.X_train, keep), whole.y_train[keep], whole.spec)
    train, held, _ = grouped_partition(dev, seed, grouping)
    audit = overlap_audit(dev, train, held)
    if audit.get("search_duplicate_rows", 0) or audit.get("search_rows_sharing_text", 0):
        raise ValueError("Overlap in outer partition")
    task = DevelopmentTask(dev.kind, "training_partition", take(dev.X_train, train),
                           dev.y_train[train], None, None, dev.spec)
    return task, take(dev.X_train, held), dev.y_train[held], dict(
        train_rows=keep[train].tolist(), held_rows=keep[held].tolist(), audit=audit)


def run_arm(task, folder, searches, finalists):
    summary = run_one(task, "local_interleaved_routes", SEARCH_SEED, folder, None,
                      **(RUN_OPTIONS | dict(max_trials=searches, confirmation_top_k=finalists)))
    fits = summary["classifier_fit_accounting"]["completed_classifier_fits"]
    valid = (summary["valid_policy_run"] and summary["refit_succeeded"]
             and summary["confirmation"]["valid_confirmation"] and summary["failed_trials"] == 0
             and summary["llm_calls"] == 0 and summary["trials_attempted"] == searches
             and fits == searches + 2 * finalists + 1 and not summary["test_accessed"])
    if not valid:
        raise ValueError(f"Incomplete history or unequal fit budget: {folder / 'summary.json'}")


def arm_record(folder, reused=False):
    summary = read(folder / "summary.json")
    model = folder / "selected_model.joblib"
    model_hash = sha(model)
    if model_hash != read(folder / "final_selection.json")["model_sha256"]:
        raise ValueError(f"Unexpected final model hash: {model}")
    history = read(folder / "trials.json")
    return dict(candidate=summary["selected_candidate"], model_file=str(model), model_sha256=model_hash,
                logical_fits=summary["classifier_fit_accounting"]["completed_classifier_fits"],
                elapsed_seconds=summary["elapsed_seconds"], reused=reused,
                path=[row["candidate"] for row in history],
                searched_routes=sorted({route_of(row["candidate"]) for row in history}),
                finalist_ids=summary["confirmation"]["shortlist"])


def replay_two_finalists(control, baseline, folder):
    history = read(control / "trials.json")
    confirmation = read(control / "confirmation.json")
    measurements = {row["candidate"]: row["measurements"] for row in confirmation["candidates"]}
    replay = decide(shortlist(history, 2), measurements, confirmation["seeds"], confirmation["grouping"])
    if not replay["valid_confirmation"]:
        raise ValueError("The reused confirmation measurements are incomplete")
    save(folder / "replay_4_2.json", replay)
    winner = replay["selected_candidate"]
    record = baseline | dict(candidate=winner, logical_fits=9, reused=True,
                             elapsed_seconds=None, finalist_ids=replay["shortlist"],
                             new_fits_in_replay=0)
    if winner != baseline["candidate"]:
        payload = joblib.load(control / "refit_data.joblib")
        payload["feature_cache_dir"] = str(folder / "feature_cache")
        refit_input = folder / "refit_data.joblib"
        joblib.dump(payload, refit_input)
        component = folder / "refitted_component.joblib"
        target = folder / "replayed_model.joblib"
        model, result = train_candidate(refit_input, winner, component, RUN_OPTIONS["trial_timeout"])
        if result["status"] != "completed" or result.get("classifier_fits", 1) != 1:
            raise ValueError(f"Replay winner could not be refit: {result}")
        joblib.dump(dict(kind=payload["kind"], candidate=winner, model=model), target)
        record.update(model_file=str(target), model_sha256=sha(target), new_fits_in_replay=1)
    return record


def predict(record, features, labels):
    path = Path(record["model_file"])
    if sha(path) != record["model_sha256"]:
        raise ValueError(f"Frozen model changed: {path}")
    artifact = joblib.load(path)
    if artifact["candidate"] != record["candidate"]:
        raise ValueError(f"Frozen model and declared candidate disagree: {path}")
    values = np.asarray(predict_with_model(artifact["kind"], artifact["model"], features)).reshape(-1)
    if values.shape != labels.shape:
        raise ValueError(f"Prediction shape mismatch: {path}")
    return values


def compare_case(folder, reused_control, task, features, labels, seed):
    folder.mkdir(parents=True)
    control = reused_control or folder / "search4_confirm3"
    expanded = folder / "search6_confirm2"
    if reused_control:
        frozen_control(control)
        run_arm(task, expanded, 6, 2)
    elif seed == SEEDS[1]:
        run_arm(task, expanded, 6, 2)
        run_arm(task, control, 4, 3)
    else:
        run_arm(task, control, 4, 3)
        run_arm(task, expanded, 6, 2)
    baseline = arm_record(control, reused=bool(reused_control))
    challenger = arm_record(expanded)
    if baseline["path"] != challenger["path"][:4]:
        raise ValueError("The shared four-step prefix did not match")
    replay = replay_two_finalists(control, baseline, folder)
    arms = {"4_3": baseline, "4_2": replay, "6_2": challenger}
    save(folder / "frozen_arms.json", arms)
    predictions = {name: predict(record, features, labels) for name, record in arms.items()}
    correct = {name: values == labels for name, values in predictions.items()}
    for name, record in arms.items():
        record.update(correct=int(correct[name].sum()), accuracy=float(correct[name].mean()))
    save(folder / "outer_predictions.json", {name: values.tolist() for name, values in predictions.items()})
    comparisons = {}
    for left, right in (("4_3", "4_2"), ("4_2", "6_2"), ("4_3", "6_2")):
        comparisons[f"{right}_minus_{left}"] = duel(correct[left], correct[right]) | dict(
            disagreeing_predictions=int((predictions[left] != predictions[right]).sum()))
    new = challenger["path"][4:]
    return dict(arms=arms, comparisons=comparisons, new_candidates=new,
                new_finalists=[candidate for candidate in new if candidate in challenger["finalist_ids"]],
                new_winner=challenger["candidate"] in new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task1", "task2", "pilot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in ("task1", "task2")}
    pilot = args.pilot.resolve()
    root = args.output.resolve()
    first = {name: prepare_task(path) for name, path in paths.items()}
    frozen_protocol = read(pilot / "pilot_protocol.json")
    if protocol(first) != frozen_protocol:
        raise ValueError("Frozen baseline archive, environment, source or split mismatch")
    cases = {}
    for seed in SEEDS:
        for name, path in paths.items():
            task, features, labels, partition = outer_partition(path, seed)
            if seed == SEEDS[0] and any(value != first[name][3][key] for key, value in partition.items()):
                raise ValueError(f"Expected to reproduce the frozen baseline partition for {name}")
            cases[name, seed] = task, features, labels, partition
    root.mkdir(parents=True, exist_ok=False)
    save(root / "protocol.json", dict(frozen_protocol=frozen_protocol, runner_sha256=sha(Path(__file__)),
         arm_options={"4_3": [4, 3], "4_2": [4, 2], "6_2": [6, 2]},
         outer_partitions={f"{name}_{seed}": case[3] for (name, seed), case in cases.items()}))
    results = []
    for (name, seed), (task, features, labels, _) in cases.items():
        print(f"Starting {name} seed {seed}", flush=True)
        fixed = pilot / name / "fixed" if seed == SEEDS[0] else None
        try:
            record = compare_case(root / name / str(seed), fixed, task, features, labels, seed)
        except (ValueError, OSError) as error:
            record = dict(error=str(error))
        results.append(dict(task=name, seed=seed, **record))
        save(root / "results.json", results)
        print(json.dumps({"task": name, "seed": seed, **(record.get("comparisons") or record)}), flush=True)
    return int(any("error" in record for record in results))


if __name__ == "__main__":
    raise SystemExit(main())
