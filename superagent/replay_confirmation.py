"""Replay the sealed selector on frozen E1 candidates, without training or API calls."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from .acceptance import check_predictions, digest, load_accepted_model
from .data import load_development_task
from .models import predict_with_model
from .runtime import take
from .selection import confirm_once, frozen_shortlist


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replay(task, run, summary, source_hash):
    manifest = read_json(run / "runtime_manifest.json")
    frozen = read_json(run / "confirmation.json")
    if (manifest["source_sha256"] != source_hash or summary["test_accessed"]
            or frozen["enabled"] or not frozen.get("reserved_for_replay")):
        raise ValueError(f"Run does not match the E1 protocol: {run.name}")
    split_path = run / "split.json"
    if digest(split_path) != frozen["split_sha256"]:
        raise ValueError(f"Split changed after search: {run.name}")
    history = read_json(run / "trials.json")
    shortlist = frozen_shortlist(history)
    expected = [(row["trial"], row["candidate"], row["search_accuracy"]) for row in shortlist]
    recorded = [(row["trial"], row["candidate"], row["search_accuracy"]) for row in frozen["shortlist"]]
    if not shortlist or expected != recorded or shortlist[0]["candidate"] != summary["selected_candidate"]:
        raise ValueError(f"Frozen shortlist disagrees with E1: {run.name}")
    for row in frozen["shortlist"]:
        if digest(run / f"trial_{row['trial']}.joblib") != row["artifact_sha256"]:
            raise ValueError(f"Frozen candidate artifact changed: {run.name}, trial {row['trial']}")
    indices = read_json(split_path)
    fit, search, sealed = (set(indices[key]) for key in
                           ("train_indices", "search_indices", "confirmation_indices"))
    if not sealed or fit & search or fit & sealed or search & sealed or (
            fit | search | sealed) != set(range(len(task.y_train))):
        raise ValueError(f"Invalid held-out confirmation partition: {run.name}")
    sealed_indices = np.asarray(indices["confirmation_indices"], dtype=int)
    started = time.perf_counter()
    selected, model, result = confirm_once(
        task.kind, shortlist, run, take(task.X_train, sealed_indices), task.y_train[sealed_indices],
        enabled=True, switch_threshold=frozen["selection_threshold"])
    confirmation_seconds = time.perf_counter() - started
    accepted, _ = load_accepted_model(run)
    if accepted["candidate"] != summary["selected_candidate"]:
        raise ValueError(f"Accepted model disagrees with the E1 summary: {run.name}")
    baseline_predictions = check_predictions(
        read_json(run / "validation_predictions.json"), len(task.y_val), accepted["labels"])
    baseline = float(accuracy_score(task.y_val, baseline_predictions))
    if baseline != summary["official_validation_accuracy"]:
        raise ValueError(f"Validation predictions disagree with the E1 summary: {run.name}")
    score = baseline
    if result["switched"]:
        predictions = check_predictions(predict_with_model(task.kind, model, task.X_val),
                                        len(task.y_val), accepted["labels"])
        score = float(accuracy_score(task.y_val, predictions))
    return {"c_off_candidate": summary["selected_candidate"], "c_on_candidate": selected["candidate"],
            "c_off_validation": baseline, "c_on_validation": score,
            "validation_delta": score - baseline, "switched": result["switched"],
            "shortlist_size": len(shortlist), "confirmation_seconds": confirmation_seconds,
            "confirmation": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    protocol = read_json(batch / "protocol.json")
    if not protocol.get("reserve_confirmation") or protocol.get("confirmation"):
        parser.error("Replay requires a C-off batch created with --reserve-confirmation.")
    rows = []
    for task_index, package in enumerate(protocol["packages"]):
        task = load_development_task(package)
        for strategy in protocol["strategies"]:
            for seed in protocol["seeds"]:
                run = batch / f"task{task_index}_{task.kind}_{strategy}_{seed}"
                summary_path = run / "summary.json"
                row = {"task_index": task_index, "task_kind": task.kind, "strategy": strategy, "seed": seed}
                if not summary_path.is_file():
                    row["status"] = "missing"
                else:
                    summary = read_json(summary_path)
                    if not summary["acceptance_passed"]:
                        row["status"] = "not_accepted"
                    else:
                        row.update(status="replayed", valid_policy_run=summary["valid_policy_run"],
                                   **replay(task, run, summary, protocol["source_sha256"]))
                rows.append(row)
    (batch / "confirmation_ablation.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = ["# E2: confirmation selector on frozen E1 trajectories", "",
             "Validation delta = C-on minus C-off for the same fitted candidates. One-candidate trajectories cannot switch.",
             "Mean gaps use accepted, valid policy runs only. Replayed fallbacks and missing runs remain in the JSON.",
             "", "| Task | Policy | Valid / planned | Two candidates | Switches | Improved / worse | Mean delta (pp) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for i, strategy in sorted({(r["task_index"], r["strategy"]) for r in rows}):
        group = [r for r in rows if (r["task_index"], r["strategy"]) == (i, strategy)]
        valid = [r for r in group if r.get("valid_policy_run")]
        deltas = [r["validation_delta"] for r in valid]
        mean = f"{100 * statistics.mean(deltas):+.3f}" if deltas else "N/A"
        lines.append(f"| {i}: {group[0]['task_kind']} | {strategy} | {len(valid)}/{len(group)} | "
                     f"{sum(r['shortlist_size'] == 2 for r in valid)} | {sum(r['switched'] for r in valid)} | "
                     f"{sum(d > 0 for d in deltas)} / {sum(d < 0 for d in deltas)} | {mean} |")
    report = batch / "confirmation_ablation.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Confirmation ablation: {report}", flush=True)


if __name__ == "__main__":
    main()
