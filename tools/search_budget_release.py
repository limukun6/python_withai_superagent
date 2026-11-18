"""Apply the replication rule and freeze both released-task models before validation."""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

import numpy as np

from superagent.adaptive_local import save, sha
from superagent.data import DevelopmentTask, load_development_task, load_training_task
from tools.expanded_feedback_pilot import prepare_task, protocol
from tools.search_budget_study import REPLICATION_SEEDS, arm_record, predict, read, run_arm


def select(results):
    expected = {(task, seed) for task in ("task1", "task2") for seed in REPLICATION_SEEDS}
    keys = [(row["task"], row["seed"]) for row in results]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Replication has not recorded all six planned pairs")
    invalid = [key for key, row in zip(keys, results) if "error" in row]
    if invalid:
        return dict(selected="4_3", reason="Incomplete challenger comparison", invalid=invalid)
    stats = {}
    gains_by_time = []
    for task in ("task1", "task2"):
        rows = [row for row in results if row["task"] == task]
        deltas = [row["comparisons"]["6_2_minus_4_3"]["delta"] for row in rows]
        time_gains = [1 - row["arms"]["6_2"]["elapsed_seconds"] / row["arms"]["4_3"]["elapsed_seconds"]
                      for row in rows]
        gains_by_time.extend(time_gains)
        stats[task] = dict(deltas=deltas, mean_delta=mean(deltas), worst_delta=min(deltas),
                           wins=sum(d > 0 for d in deltas), ties=deltas.count(0), losses=sum(d < 0 for d in deltas),
                           faster=sum(t > 0 for t in time_gains), median_relative_time_saved=median(time_gains))
    safe = all(row["mean_delta"] >= -.002 and row["worst_delta"] >= -.01 for row in stats.values())
    positive = [row for row in stats.values() if row["mean_delta"] >= .002]
    accuracy = safe and bool(positive) and all(row["wins"] >= 2 for row in positive)
    speed = safe and median(gains_by_time) >= .20 and all(row["faster"] >= 2 for row in stats.values())
    return dict(selected="6_2" if accuracy or speed else "4_3",
                meets_accuracy_rule=accuracy, meets_efficiency_rule=speed,
                within_loss_limits=safe, tasks=stats, median_relative_time_saved=median(gains_by_time))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task1", "task2", "replication", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    inputs = {name: getattr(args, name).resolve() for name in ("task1", "task2")}
    replication = args.replication.resolve()
    reproduction = protocol({name: prepare_task(path) for name, path in inputs.items()})
    frozen_protocol = read(replication / "protocol.json")
    if (frozen_protocol.get("stage") != "replication"
            or frozen_protocol.get("frozen_protocol") != reproduction
            or frozen_protocol.get("runner_sha256") != sha(Path(__file__).with_name("search_budget_study.py"))
            or tuple(frozen_protocol.get("seeds", [])) != REPLICATION_SEEDS):
        raise ValueError("Replication artifacts do not match the frozen code, environment and data")
    results_path = replication / "results.json"
    selection = select(read(results_path))
    searches, finalists = map(int, selection["selected"].split("_"))
    root.mkdir(parents=True, exist_ok=False)
    save(root / "release_protocol.json", dict(selection=selection, replication_sha256=sha(results_path),
         runner_sha256=sha(Path(__file__)),
         source=reproduction["source"], dependencies=reproduction["dependencies"],
         archives={name: sha(path) for name, path in inputs.items()}, selection_data="training_partitions_only",
         evaluation="provided_validation", model_api_calls=0))
    print(f"Selected {searches} searches and {finalists} confirmation finalists", flush=True)
    arms = {}
    for name, archive in inputs.items():
        loaded = load_training_task(archive)
        task = DevelopmentTask(loaded.kind, "released_training_only", loaded.X_train, loaded.y_train,
                               None, None, loaded.spec)
        run_arm(task, root / name, searches, finalists)
        arms[name] = arm_record(root / name)
        arms[name]["train_rows"] = len(task.y_train)
        print(f"{name} frozen: {arms[name]['candidate']}", flush=True)
    save(root / "frozen_models.json", arms)
    scores = {}
    for name, archive in inputs.items():
        task = load_development_task(archive)
        if task.X_val is None or task.y_val is None:
            raise ValueError(f"Missing released validation data: {name}")
        values = predict(arms[name], task.X_val, task.y_val)
        label_domain = set(np.unique(task.y_train).tolist())
        if not set(values.tolist()) <= label_domain:
            raise ValueError(f"Predictions contain an undeclared label: {name}")
        correct = int((values == task.y_val).sum())
        predictions = root / name / "validation_predictions.csv"
        with predictions.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["row_id", "prediction"])
            writer.writerows(enumerate(values.tolist()))
        scores[name] = dict(candidate=arms[name]["candidate"], rows=len(values), correct=correct,
                            accuracy=correct / len(values), predictions_sha256=sha(predictions))
        print(scores[name], flush=True)
    save(root / "validation_scores.json", scores)


if __name__ == "__main__":
    main()
