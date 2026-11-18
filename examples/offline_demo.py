"""Exercise real training/export with a scripted planner; no API or LLM calls.

This is an integration demonstration, not an architecture-performance result.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import numpy as np
from superagent.benchmark import run_one
from superagent.data import load_development_task


class ScriptedPlanner:
    """Fixture that makes outcomes visible/hidden and records zero API usage."""
    def __init__(self):
        self.requests = []

    def ask(self, payload, logger, role):
        self.requests.append(payload)
        available = {candidate["id"] for candidate in payload["available_candidates"]}
        observations = payload["observations"]
        low_score = bool(observations and observations[-1].get("search_accuracy", 1) < .75)
        order = ("logistic_C4.0", "logistic_C1.0", "logistic_C0.25") if low_score else (
            "logistic_C0.25", "logistic_C1.0", "logistic_C4.0")
        chosen = next(candidate for candidate in order if candidate in available)
        previous = payload["plans"][-1]["trial"]
        logger.usage(event="scripted_planner_call", role=role, usage_known=True,
                     input_tokens=0, output_tokens=0, cost_rmb=0.0,
                     course_cost_rmb=0.0, actual_cost_rmb=0.0,
                     note="OFFLINE FIXTURE: no model invocation or billable usage")
        return {
            "candidate": chosen,
            "reason": "Offline fixture selecting an available logistic configuration.",
            "hypothesis": "Changing regularization will improve search accuracy by at least 0.01.",
            "review": ("The latest expectation check is available in observations; inspect it before the next choice."
                       if payload["feedback_visible"] else "Observed outcomes are hidden; earlier expectations remain unverified to this planner."),
            "evidence_trials": [previous] if payload["feedback_visible"] else [],
            "expected_change": {"metric": "search_accuracy", "reference_trial": previous,
                                "direction": "increase", "min_delta": .01},
        }


def create_package(path):
    with zipfile.ZipFile(path, "w") as archive:
        for split, count, offset in (("train", 120, 0), ("val", 24, 1000), ("test", 12, 2000)):
            labels = np.asarray([i % 2 for i in range(count)])
            features = np.asarray([["blue" if y else "red", str(offset + i)] for i, y in enumerate(labels)])
            for name, array in ((f"C_{split}.npy", features), (f"y_{split}.npy", labels)):
                if split == "test" and name.startswith("y_"):
                    # Any attempt to load test labels would fail loudly.
                    archive.writestr(name, b"POISON TEST LABELS: must not load")
                else:
                    stream = io.BytesIO()
                    np.save(stream, array)
                    archive.writestr(name, stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/offline_reflective_demo")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    package = root / "synthetic_task.zip"
    create_package(package)
    task = load_development_task(package)
    summaries = []
    for strategy in ("reflective", "reflective_no_feedback"):
        client = ScriptedPlanner()
        with patch("requests.post", side_effect=AssertionError("Network is disabled in the offline example")):
            summary = run_one(task, strategy, 11, root / strategy, client,
                              max_trials=3, time_limit=60, trial_timeout=20, anchor_first=True)
        if not summary["completed"] or not summary["valid_policy_run"]:
            raise RuntimeError(f"Synthetic integration did not complete: {summary}")
        destination = root / strategy / "predictions.csv"
        command = [sys.executable, "-m", "superagent", "predict", str(package),
                   "--run-dir", str(root / strategy), "--output", str(destination)]
        prediction = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, check=True)
        if len(destination.read_text().splitlines()) != 13:
            raise AssertionError("Prediction row count mismatch")
        summaries.append({
            "strategy": strategy, "planner_kind": "scripted_fixture_not_llm",
            "scripted_decisions": len(client.requests), "real_api_calls": 0,
            "trials_attempted": summary["trials_attempted"], "training_completed": summary["completed"],
            "hypotheses_evaluated": summary["hypotheses_evaluated"],
            "expectations_met": summary["expectations_met"],
            "fresh_process_prediction_completed": prediction.returncode == 0,
            "prediction_count": 12,
        })
    report = {"purpose": "synthetic integration only; not a model-quality or architecture comparison",
              "source_base": "68e6772daa9ced52849c16ab694fefb88029a860",
              "real_api_calls": 0, "summaries": summaries}
    (root / "demo_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
