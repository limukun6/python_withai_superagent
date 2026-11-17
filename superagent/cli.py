from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from .data import load_development_task
from .events import RunLogger
from .planner import make_planner
from .predict import create_predictions
from .profile import profile
from .runner import run_portfolio, save_selected_model
from .runtime import run_task, STRATEGIES
from .experiment_llm import ExperimentLLM


def _legacy_run(args: argparse.Namespace) -> int:
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


def _run(args: argparse.Namespace) -> int:
    task = load_development_task(args.task_package)
    client = None
    if args.strategy not in ("deterministic", "random"):
        if not args.api_key_file:
            raise ValueError("LLM strategies require --api-key-file.")
        client = ExperimentLLM(args.api_url, args.api_key_file, args.model, args.llm_budget_rmb)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(args.output_dir).resolve() / stamp
    summary = run_task(task, args.strategy, args.seed, directory, client,
                       args.max_trials, args.time_limit, args.trial_timeout, args.confirmation)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["completed"] else 1


def _predict(args: argparse.Namespace) -> int:
    destination = create_predictions(args.task_package, args.run_dir, args.output)
    print(f"Predictions written to: {destination}")
    print("Test labels were not read or evaluated.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-driven classification SuperAgent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Run the same audited task core used by benchmark")
    run.add_argument("task_package", help="Path to a supported task zip archive")
    run.add_argument("--output-dir", default="runs", help="Directory for reproducible run artifacts")
    run.add_argument("--strategy", choices=STRATEGIES, default="deterministic")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--max-trials", type=int, default=3)
    run.add_argument("--time-limit", type=float, default=1200)
    run.add_argument("--trial-timeout", type=float, default=120)
    run.add_argument("--confirmation", action="store_true", help="Use a sealed third inner partition for one-time top-two selection")
    run.add_argument("--api-url", default="https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions")
    run.add_argument("--api-key-file", help="UTF-8 API key file; never written to run artifacts")
    run.add_argument("--model", default="DeepSeek-V4-Pro")
    run.add_argument("--llm-budget-rmb", type=float, default=2.0)
    run.set_defaults(handler=_run)
    legacy = subcommands.add_parser("legacy-run", help="Historical exhaustive baseline; not the current experiment core")
    legacy.add_argument("task_package")
    legacy.add_argument("--output-dir", default="runs/legacy")
    legacy.add_argument("--planner", choices=["local", "llm"], default="local")
    legacy.add_argument("--api-url")
    legacy.add_argument("--api-key-file")
    legacy.add_argument("--model")
    legacy.add_argument("--llm-budget-rmb", type=float, default=20.0)
    legacy.set_defaults(handler=_legacy_run)
    predict = subcommands.add_parser("predict", help="Use a frozen selected model to write test predictions without reading labels")
    predict.add_argument("task_package", help="Original task zip archive")
    predict.add_argument("--run-dir", required=True, help="Completed development run directory")
    predict.add_argument("--output", required=True, help="Destination CSV for row IDs and predictions")
    predict.set_defaults(handler=_predict)
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
