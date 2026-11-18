"""Measure search-score feedback and shortlist width on matched training holdouts."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.confirmation import shortlist
from superagent.contracts import ExperimentProposal
from superagent.evidence_agent import route_of
from superagent.evidence_schedule import catalog_for, duel
from superagent.local_search import LocalSearchPolicy
from superagent.metrics import completed, score_of
from tools.expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED, prepare_task, protocol
from tools.search_budget_study import arm_record, outer_partition, predict, read


SEEDS = {"development": (43111, 43117), "replication": (43141, 43153, 43167)}
ORDERS = ("BFK", "FKB", "KBF")
STRATEGY = "local_interleaved_routes"


class MaskedSearch:
    """Replace scores only at the local selector boundary."""

    def __init__(self, catalog, seed):
        self.policy = LocalSearchPolicy("interleaved_routes", catalog, seed)

    def propose(self, state):
        record = state.record()
        history = record["history"]
        visible = [
            dict(candidate=row["candidate"], route=route_of(row["candidate"]),
                 status=row["status"], primary_metric=row["primary_metric"],
                 search_score=0.5 if completed(row) else None)
            for row in history
        ]
        chosen, reason = self.policy.choose(visible)
        config = next(c for c in record["available_candidates"] if c["id"] == chosen)
        return ExperimentProposal.from_decision(record["next_trial"], config, reason, None, history)


def atomic_save(path, value):
    temporary = path.with_suffix(".tmp")
    save(temporary, value)
    temporary.replace(path)


def fit_arm(task, folder, arm):
    finalists = 2 if arm == "K" else 3
    masked = MaskedSearch(catalog_for(task.kind), SEARCH_SEED) if arm == "F" else None
    started = datetime.now(timezone.utc).isoformat()
    summary = run_one(task, STRATEGY, SEARCH_SEED, folder, None, decision_policy=masked,
                      **(RUN_OPTIONS | dict(confirmation_top_k=finalists)))
    history = read(folder / "trials.json")
    expected_shortlist = [row["candidate"] for row in shortlist(history, finalists)]
    confirmation = summary["confirmation"]
    usage = {key: summary[key] for key in ("llm_calls", "tokens", "course_cost_rmb", "usage_complete")}
    valid = (summary["valid_policy_run"] and summary["refit_succeeded"] and confirmation["valid_confirmation"]
             and summary["failed_trials"] == 0 and summary["trials_attempted"] == 4
             and usage == dict(llm_calls=0, tokens=0, course_cost_rmb=0, usage_complete=True)
             and summary["classifier_fit_accounting"]["completed_classifier_fits"] == 5 + 2 * finalists
             and not summary["test_accessed"])
    record = dict(status="valid" if valid else "incomplete", summary_file=str(folder / "summary.json"),
                  known_completed_fits=summary["classifier_fit_accounting"]["completed_classifier_fits"],
                  usage=usage, score_masked_for_search=masked is not None,
                  started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat())
    if valid:
        if confirmation["shortlist"] != expected_shortlist:
            raise ValueError(f"Confirmation did not use measured scores in {folder}")
        for candidate in confirmation["candidates"]:
            search = next(row for row in history if row["candidate"] == candidate["candidate"])
            primary = next(row for row in candidate["measurements"] if row["seed"] == SEARCH_SEED)
            if score_of(primary) != score_of(search):
                raise ValueError(f"Primary confirmation score must match the true search score in {folder}")
        record.update(arm_record(folder), search_scores=[score_of(row) for row in history])
    return record


def compare_case(folder, case, order):
    task, features, labels, partition = case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").is_file():
        record = read(folder / "result.json")
        if tuple(record["arms"]) != tuple(order):
            raise ValueError(f"Saved arms differ from the protocol in {folder}")
        for arm in record["arms"].values():
            if arm["status"] == "valid" and sha(Path(arm["model_file"])) != arm["model_sha256"]:
                raise ValueError(f"Saved model fingerprint differs in {folder}")
        if record.get("prediction_sha256") and sha(Path(record["prediction_file"])) != record["prediction_sha256"]:
            raise ValueError(f"Saved prediction fingerprint differs in {folder}")
        return record
    attempt = folder / f"attempt_{len(list(folder.glob('attempt_*'))) + 1:02}"
    attempt.mkdir()
    arms = {}
    for arm in order:
        print(f"Starting {folder.parent.name} seed {folder.name} arm {arm}", flush=True)
        arms[arm] = fit_arm(task, attempt / arm, arm)
        atomic_save(attempt / "frozen_arms.json", arms)
    if {"B", "K"} <= arms.keys() and all(arms[x]["status"] == "valid" for x in "BK"):
        if any(arms["B"][key] != arms["K"][key] for key in ("path", "search_scores")):
            raise ValueError(f"B and K must share the complete search trajectory in {folder}")
    values = {name: predict(arm, features, labels) for name, arm in arms.items() if arm["status"] == "valid"}
    correct = {name: prediction == labels for name, prediction in values.items()}
    for name, mask in correct.items():
        arms[name].update(correct=int(mask.sum()), accuracy=float(mask.mean()))
    prediction_file = attempt / "outer_predictions.json"
    atomic_save(prediction_file, dict(row_ids=partition["held_rows"], labels=labels.tolist(),
                predictions={name: prediction.tolist() for name, prediction in values.items()}))
    pairs = {}
    for key, left, right in (("B_minus_F", "F", "B"), ("K_minus_B", "B", "K")):
        if left not in values or right not in values:
            continue
        seconds = {name: arms[name]["elapsed_seconds"] for name in (left, right)}
        pairs[key] = dict(
            accuracy_delta=arms[right]["accuracy"] - arms[left]["accuracy"],
            time_saved_K_vs_B=(seconds[left] - seconds[right]) / seconds[left] if right == "K" else None,
            first_action_difference=next((i for i, (a, b) in enumerate(zip(arms[left]["path"], arms[right]["path"]), 1)
                                          if a != b), None),
            disagreeing_predictions=int((values[left] != values[right]).sum()),
            **duel(correct[left], correct[right]),
        )
    if "B_minus_F" in pairs and arms["B"]["path"][:2] != arms["F"]["path"][:2]:
        raise ValueError(f"B and F must share the anchor and route exploration step in {folder}")
    result = dict(arms=arms, pairs=pairs, rows=len(labels), prediction_file=str(prediction_file),
                  prediction_sha256=sha(prediction_file))
    atomic_save(folder / "result.json", result)
    return result


def assess(results, pair, stage):
    required = {"B", "F" if pair == "B_minus_F" else "K"}
    planned = [row for row in results if required <= row["arms"].keys()]
    valid = [row for row in planned if pair in row["pairs"]]
    by_task = {task: [row["pairs"][pair] for row in valid if row["task"] == task]
               for task in sorted({row["task"] for row in planned})}
    averages = {task: mean(x["accuracy_delta"] for x in rows) for task, rows in by_task.items() if rows}
    all_pairs = [row["pairs"][pair] for row in valid]
    complete = bool(planned) and len(valid) == len(planned)
    safe = (complete and min(averages.values()) >= -0.002 - 1e-12
            and min(x["accuracy_delta"] for x in all_pairs) >= -0.01 - 1e-12)
    if pair == "B_minus_F":
        worth_repeating = any(x["first_action_difference"] is not None for x in all_pairs)
        passed = safe and any(averages[task] >= 0.002 - 1e-12 and
                              sum(x["accuracy_delta"] > 0 for x in rows) >= 2
                              for task, rows in by_task.items())
        return dict(valid=len(valid), planned=len(planned), average_accuracy_delta=averages,
                    changed_paths=sum(x["first_action_difference"] is not None for x in all_pairs),
                    replicate=complete and worth_repeating, demonstrates_improvement=stage == "replication" and passed)
    saved = median(x["time_saved_K_vs_B"] for x in all_pairs) if all_pairs else None
    wins = {task: sum(x["time_saved_K_vs_B"] > 0 for x in rows) for task, rows in by_task.items()}
    meets_gate = safe and saved >= 0.2 - 1e-12
    return dict(valid=len(valid), planned=len(planned), average_accuracy_delta=averages,
                median_time_saved=saved, faster_pairs=wins, replicate=meets_gate,
                eligible=stage == "replication" and meets_gate and all(count >= 2 for count in wins.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("task1", "task2", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    args = parser.parse_args()
    archives = {name: getattr(args, name).resolve() for name in ("task1", "task2")}
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    arms = "BFK"
    if args.stage == "replication":
        gate = read(root / "development" / "decision.json")
        arms = "".join(["B", "F" if gate["B_minus_F"]["replicate"] else "",
                        "K" if gate["K_minus_B"]["replicate"] else ""])
        if arms == "B":
            print("No comparison entered replication.", flush=True)
            return
    stage = root / args.stage
    stage.mkdir(exist_ok=True)
    setup = protocol({name: prepare_task(path) for name, path in archives.items()})
    identity = dict(environment=setup, study_sha256=sha(Path(__file__)),
                    outer_partition_helper_sha256=sha(Path(__file__).with_name("search_budget_study.py")))
    if args.stage == "replication" and identity != read(root / "development" / "protocol.json")["identity"]:
        raise ValueError("Development and replication must use the same code, inputs, and dependencies")
    cases, order = {}, {}
    for index, seed in enumerate(SEEDS[args.stage]):
        for offset, (name, archive) in enumerate(archives.items()):
            case_id = f"{name}_{seed}"
            cases[case_id] = outer_partition(archive, seed)
            order[case_id] = "".join(x for x in ORDERS[(index + offset) % len(ORDERS)] if x in arms)
    frozen = dict(identity=identity, run_order=order, outer_partitions={k: v[3] for k, v in cases.items()})
    manifest = stage / "protocol.json"
    if manifest.exists() and read(manifest) != frozen:
        raise ValueError(f"Cannot resume after protocol changed: {manifest}")
    if not manifest.exists():
        atomic_save(manifest, frozen)
    results = []
    for case_id, case in cases.items():
        name, seed = case_id.split("_")
        record = compare_case(stage / name / seed, case, order[case_id])
        results.append(dict(task=name, seed=int(seed), **record))
        atomic_save(stage / "results.json", results)
        print(f"Completed {case_id} {record['pairs']}", flush=True)
    decision = {pair: assess(results, pair, args.stage) for pair in ("B_minus_F", "K_minus_B")}
    atomic_save(stage / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
