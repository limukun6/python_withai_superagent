from __future__ import annotations

import argparse
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn

from .data import load_development_task
from .events import RunLogger
from .planner import make_planner
from .predict import create_predictions
from .profile import profile
from .runner import run_portfolio, save_selected_model


def _run(args: argparse.Namespace) -> int:
    task = load_development_task(args.task_package)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{task.kind}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(run_dir)
    logger.initialize_usage_ledger()
    logger.event("run_started", source=task.source, mode="development", test_accessed=False)
    logger.write_json("task_spec.json", task.spec)
    data_profile = profile(task)
    logger.write_json("data_profile.json", data_profile)
    logger.write_json("run_manifest.json", {
        "mode": "development",
        "test_accessed": False,
        "random_seed": 42,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    })
    planner = make_planner(args, logger)
    model, selection, results = run_portfolio(task, logger, data_profile, planner)
    logger.write_experiments(results)
    save_selected_model(model, selection, run_dir, logger)
    print(f"Selected {selection['family']} with validation accuracy {selection['validation_accuracy']:.4f}")
    print(f"Run artifacts: {run_dir.resolve()}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    destination = create_predictions(args.task_package, args.run_dir, args.output)
    print(f"Predictions written to: {destination}")
    print("Test labels were not read or evaluated.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-driven classification SuperAgent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Run development experiments without reading test data")
    run.add_argument("task_package", help="Path to a supported task zip archive")
    run.add_argument("--output-dir", default="runs", help="Directory for reproducible run artifacts")
    run.add_argument("--planner", choices=["local", "llm"], default="local", help="Experiment decision policy; local costs nothing")
    run.add_argument("--api-url", help="OpenAI-compatible chat-completions URL for --planner llm")
    run.add_argument("--api-key-file", help="UTF-8 API key file for --planner llm")
    run.add_argument("--model", help="Provider model ID for --planner llm")
    run.set_defaults(handler=_run)
    predict = subcommands.add_parser("predict", help="Use a frozen selected model to write test predictions without reading labels")
    predict.add_argument("task_package", help="Original task zip archive")
    predict.add_argument("--run-dir", required=True, help="Completed development run directory")
    predict.add_argument("--output", required=True, help="Destination CSV for row IDs and predictions")
    predict.set_defaults(handler=_predict)
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
