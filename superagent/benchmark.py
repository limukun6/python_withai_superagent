"""Controlled architecture comparisons using the shared task runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import load_development_task
from .experiment_llm import ExperimentLLM
from .runtime import run_task as run_one, Policy, inner_split, STRATEGIES, RUNTIME_VERSION


def write_report(destination, rows):
    lines = ["# Agent architecture comparison (development)", "",
        "Official validation is scored only after inner-search selection. Previously used official validation is not a pristine holdout.",
        "Only valid policy runs enter accuracy averages; fallback runs remain visible in summaries. Prices are provisional platform estimates.", "",
        "| Task | Policy | Valid / attempted | Mean validation accuracy | Sample SD | Mean seconds (all) | Tokens (all) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for task, strategy in sorted({(r["task_kind"], r["strategy"]) for r in rows}):
        group = [r for r in rows if r["task_kind"] == task and r["strategy"] == strategy]
        scores = [r["official_validation_accuracy"] for r in group if r["valid_policy_run"]]
        mean = f"{statistics.mean(scores):.4f}" if scores else "N/A"
        sd = f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "N/A"
        lines.append(f"| {task} | {strategy} | {len(scores)} / {len(group)} | {mean} | {sd} | {statistics.mean(r['elapsed_seconds'] for r in group):.1f} | {sum(r['tokens'] for r in group)} |")
    lines += ["", "## Interpretation limits", "",
        "One repeat is a feasibility pilot, not evidence of architecture superiority. Compare matched seeds, report all failures, and repeat before selecting an architecture.",
        "Reviewed agents use extra calls; this trial-matched comparison does not isolate architecture from token expenditure. A token-matched follow-up is required.",
        "The fixed search space contains 14 configurations per task. These experiments test search policy, not unrestricted code generation or generality to unseen modalities.",
        "All classifiers retain model seed 42; repeat seeds control the shared data split and random search. This does not estimate independent model-training variance."]
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--trial-timeout", type=float, default=120)
    parser.add_argument("--output-dir", default="runs/benchmarks")
    parser.add_argument("--api-url", default="https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions")
    parser.add_argument("--api-key-file")
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--batch-budget-rmb", type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.max_trials <= 14 or not 0 < args.time_limit < float("inf") or not 0 < args.trial_timeout < float("inf"):
        parser.error("Use 1..14 trials and positive finite time limits.")
    strategies = list(dict.fromkeys(args.strategies))
    seeds = list(dict.fromkeys(args.seeds))
    client = None
    if any(s not in ("random", "deterministic") for s in strategies):
        if not args.api_key_file:
            parser.error("LLM strategies require --api-key-file.")
        client = ExperimentLLM(args.api_url, args.api_key_file, args.model, args.batch_budget_rmb)
    destination = Path(args.output_dir).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    config = {k: v for k, v in vars(args).items() if k != "api_key_file"}
    config.update({"source_sha256": source_hash.hexdigest(), "python": sys.version, "runtime_version": RUNTIME_VERSION,
                   "price_status": "provisional_platform_price", "test_accessed": False})
    (destination / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for task_index, package in enumerate(args.packages):
        task = load_development_task(package)
        for seed in seeds:
            order = strategies.copy()
            random.Random(seed).shuffle(order)
            for strategy in order:
                directory = destination / f"task{task_index}_{task.kind}_{strategy}_{seed}"
                summary = run_one(task, strategy, seed, directory, client, args.max_trials, args.time_limit, args.trial_timeout)
                rows.append(summary)
                (destination / "summaries.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                write_report(destination, rows)
                print(json.dumps(summary), flush=True)
    print(f"Comparison: {destination / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
