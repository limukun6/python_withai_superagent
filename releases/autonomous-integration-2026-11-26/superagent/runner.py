"""Autonomous local experiment loop with deterministic selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score

from .data import DevelopmentTask
from .events import RunLogger
from .models import candidates_for
from .planner import LocalPlanner, Planner


def run_portfolio(
    task: DevelopmentTask, logger: RunLogger, data_profile: dict[str, Any], planner: Planner | None = None
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    planner = planner or LocalPlanner()
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_row: dict[str, Any] | None = None
    remaining = {candidate.family: candidate for candidate in candidates_for(task.kind)}
    for index in range(1, len(remaining) + 1):
        available = list(remaining)
        try:
            chosen_family, planner_reason = planner.choose(data_profile, results, available)
        except Exception as error:
            chosen_family, planner_reason = LocalPlanner().choose(data_profile, results, available)
            logger.event(
                "planner_fallback", error=f"{type(error).__name__}: {error}",
                selected_candidate=chosen_family, fallback_reason=planner_reason,
            )
        candidate = remaining.pop(chosen_family)
        experiment_id = f"exp_{index:03d}"
        logger.event(
            "experiment_planned", experiment_id=experiment_id, family=candidate.family,
            decision_basis=planner_reason, params=candidate.params,
        )
        started = time.perf_counter()
        try:
            model, predictions = candidate.fit_predict(task.X_train, task.y_train, task.X_val)
            score = float(accuracy_score(task.y_val, predictions))
            row = {
                "experiment_id": experiment_id,
                "family": candidate.family,
                "status": "completed",
                "validation_accuracy": score,
                "train_seconds": round(time.perf_counter() - started, 3),
                "reason": "completed",
                "params": candidate.params,
            }
            logger.event("experiment_completed", **row)
            if best_row is None or score > best_row["validation_accuracy"]:
                best_model, best_row = model, row
                logger.event(
                    "incumbent_updated", experiment_id=experiment_id, validation_accuracy=score,
                    decision_basis="Highest validation accuracy; ties retain earlier simpler candidate.",
                )
        except Exception as error:  # recorded so later repair policies can reason over failures
            row = {
                "experiment_id": experiment_id,
                "family": candidate.family,
                "status": "failed",
                "validation_accuracy": "",
                "train_seconds": round(time.perf_counter() - started, 3),
                "reason": f"{type(error).__name__}: {error}",
                "params": candidate.params,
            }
            logger.event("experiment_failed", **row)
        results.append(row)
    if best_model is None or best_row is None:
        raise RuntimeError("No candidate completed successfully.")
    logger.event(
        "run_stopped", reason="All schema-appropriate portfolio candidates were evaluated.",
        selected_experiment=best_row["experiment_id"], validation_accuracy=best_row["validation_accuracy"],
    )
    return best_model, best_row, results


def save_selected_model(model: Any, row: dict[str, Any], run_dir: Path, logger: RunLogger) -> None:
    destination = run_dir / "final_model.joblib"
    joblib.dump(model, destination)
    logger.write_json("final_selection.json", {
        "selected_experiment": row["experiment_id"],
        "family": row["family"],
        "validation_accuracy": row["validation_accuracy"],
        "selection_rule": "Maximum validation accuracy; ties retain earlier portfolio candidate.",
        "model_file": destination.name,
        "test_accessed": False,
    })

