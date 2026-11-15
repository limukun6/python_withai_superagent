"""Controlled comparisons of random, one-shot, feedback and reviewed agents."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .data import DevelopmentTask, load_development_task
from .events import RunLogger
from .experiment_llm import ExperimentLLM
from .models import search_candidates, predict_with_model
from .profile import profile


def take(rows, indices):
    return rows[indices] if isinstance(rows, np.ndarray) else [rows[int(i)] for i in indices]


def inner_split(task, seed):
    # Identical rows (or reversed text pairs) stay together across the inner split.
    keys = [tuple(sorted(row)) if task.kind == "pair_text" else tuple(map(str, row)) for row in task.X_train]
    groups = np.asarray([hashlib.sha256(repr(key).encode()).hexdigest() for key in keys])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, search = next(splitter.split(np.zeros(len(task.y_train)), task.y_train, groups))
    return train, search


class Policy:
    def __init__(self, strategy, seed, client, logger, limit):
        self.strategy, self.client, self.logger, self.limit = strategy, client, logger, limit
        self.rng = random.Random(seed)
        self.plan = None

    def choose(self, data_profile, history, available):
        ids = [item["id"] for item in available]
        if self.strategy == "random":
            return self.rng.choice(ids), "Uniform random draw without replacement."
        context = {"profile": data_profile, "available_candidates": available, "max_trials": self.limit}
        if self.strategy == "oneshot":
            if self.plan is None:
                answer = self.client.ask({**context,
                    "instruction": "Return {plan: [{candidate: id, reason: explanation}]} with exactly max_trials distinct candidates, in execution order. You will receive no experiment feedback."},
                    self.logger, "initial_plan")
                plan = answer.get("plan")
                if not isinstance(plan, list) or len(plan) != self.limit:
                    raise ValueError("One-shot plan must match the trial limit.")
                selected = [item.get("candidate") for item in plan if isinstance(item, dict)]
                if len(selected) != self.limit or len(set(selected)) != self.limit or not set(selected) <= set(ids):
                    raise ValueError("Invalid or duplicate candidates in one-shot plan.")
                self.plan = list(plan)
            answer = self.plan.pop(0)
        else:
            context["completed_experiments"] = history
            answer = self.client.ask({**context,
                "instruction": "Choose the next experiment using observed metrics and failures. Return {candidate: id, reason: a testable hypothesis and why this configuration helps}."},
                self.logger, "planner")
            if self.strategy == "reviewed":
                answer = self.client.ask({**context, "proposal": answer,
                    "instruction": "Review the proposal against observed results. Keep it or select a better available candidate. Return {candidate: id, reason: critique and final hypothesis}."},
                    self.logger, "reviewer")
        if answer.get("candidate") not in ids or not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
            raise ValueError("Invalid decision schema or unavailable candidate.")
        return answer["candidate"], answer["reason"]


def train_candidate(payload_path, candidate, destination, timeout):
    started = time.monotonic()
    command = [sys.executable, "-m", "superagent.experiment_worker", str(payload_path), candidate, str(destination)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                cwd=Path(__file__).resolve().parents[1])
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout", "error": "Training time limit reached", "training_seconds": time.monotonic() - started}
    if result.returncode:
        return None, {"status": "failed", "error": result.stderr[-3000:], "training_seconds": time.monotonic() - started}
    artifact = joblib.load(destination)
    return artifact["model"], {"status": "completed", **artifact["metrics"]}


def run_one(task, strategy, seed, directory, client, max_trials=3, time_limit=600, trial_timeout=120):
    directory.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(directory)
    logger.initialize_usage_ledger()
    start = time.monotonic()
    train_idx, search_idx = inner_split(task, seed)
    logger.write_json("split.json", {"seed": seed, "train_indices": train_idx.tolist(), "search_indices": search_idx.tolist()})
    payload = {"kind": task.kind, "X_train": take(task.X_train, train_idx), "y_train": task.y_train[train_idx],
               "X_search": take(task.X_train, search_idx), "y_search": task.y_train[search_idx]}
    payload_path = directory / "inner_data.joblib"
    joblib.dump(payload, payload_path)
    inner = DevelopmentTask(task.kind, "inner_split", payload["X_train"], payload["y_train"],
                            payload["X_search"], payload["y_search"], {})
    data_profile = profile(inner)
    logger.write_json("data_profile.json", data_profile)
    available = [{"id": c.family, "params": c.params} for c in search_candidates(task.kind)]
    policy = Policy(strategy, seed, client, logger, max_trials)
    history, best_model, best_id, best_score = [], None, None, -1.0
    fallbacks, blocked_llm = 0, False
    random_fallback = random.Random(seed)
    for index in range(max_trials):
        if time.monotonic() - start >= time_limit:
            break
        try:
            if blocked_llm:
                raise RuntimeError("Planner disabled after earlier failure.")
            chosen, reason = policy.choose(data_profile, history, available)
        except Exception as error:
            blocked_llm = True
            fallbacks += 1
            chosen = random_fallback.choice(available)["id"]
            reason = "Random recovery after planner failure; this is not a valid pure LLM run."
            logger.event("planner_fallback", error=str(error), chosen=chosen)
        remaining_seconds = time_limit - (time.monotonic() - start)
        if remaining_seconds <= 0:
            break
        configuration = next(item for item in available if item["id"] == chosen)
        available = [item for item in available if item["id"] != chosen]
        logger.event("decision", trial=index + 1, candidate=chosen, reason=reason, params=configuration["params"])
        model, metrics = train_candidate(payload_path, chosen, directory / f"trial_{index+1}.joblib",
                                         min(trial_timeout, remaining_seconds))
        row = {"trial": index + 1, "candidate": chosen, "params": configuration["params"], **metrics}
        if model is not None and (metrics["search_accuracy"] > best_score or
            (metrics["search_accuracy"] == best_score and chosen < best_id)):
            best_score, best_id, best_model = metrics["search_accuracy"], chosen, model
        row["best_search_accuracy"] = None if best_model is None else best_score
        row["elapsed_seconds"] = time.monotonic() - start
        history.append(row)
        logger.event("trial_result", **row)
        logger.write_json("trials.json", history)
    # Freeze before official validation is evaluated. No refit: all designs use the same protocol.
    logger.write_json("selection.json", {"candidate": best_id, "search_accuracy": best_score if best_model else None,
                      "selection_rule": "max inner accuracy; ties by candidate id", "refit": False})
    official_accuracy = None
    if best_model is not None:
        joblib.dump({"kind": task.kind, "candidate": best_id, "model": best_model}, directory / "selected_model.joblib")
        predictions = predict_with_model(task.kind, best_model, task.X_val)
        official_accuracy = float(accuracy_score(task.y_val, predictions))
        logger.write_json("validation_predictions.json", predictions.tolist())
    usage = [json.loads(line) for line in logger.usage_path.read_text(encoding="utf-8").splitlines()]
    summary = {"strategy": strategy, "seed": seed, "task_kind": task.kind,
        "selected_candidate": best_id, "search_accuracy": best_score if best_model else None,
        "official_validation_accuracy": official_accuracy,
        "completed": best_model is not None, "valid_policy_run": best_model is not None and fallbacks == 0,
        "fallback_count": fallbacks, "trials_attempted": len(history),
        "failed_trials": sum(r["status"] != "completed" for r in history),
        "elapsed_seconds": time.monotonic() - start,
        "llm_calls": sum(u["event"] != "ledger_initialized" for u in usage),
        "tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage),
        "estimated_cost_rmb": sum(u.get("cost_rmb") or 0 for u in usage),
        "usage_complete": not any(u.get("usage_known") is False for u in usage),
        "test_accessed": False, "run_directory": str(directory)}
    logger.write_json("summary.json", summary)
    return summary


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
    parser.add_argument("--strategies", nargs="+", choices=["random", "oneshot", "feedback", "reviewed"], default=["random"])
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
    if not 1 <= args.max_trials < 14 or not 0 < args.time_limit < float("inf") or not 0 < args.trial_timeout < float("inf"):
        parser.error("Use 1..13 trials and positive finite time limits.")
    strategies = list(dict.fromkeys(args.strategies))
    seeds = list(dict.fromkeys(args.seeds))
    client = None
    if any(s != "random" for s in strategies):
        if not args.api_key_file:
            parser.error("LLM strategies require --api-key-file.")
        client = ExperimentLLM(args.api_url, args.api_key_file, args.model, args.batch_budget_rmb)
    destination = Path(args.output_dir).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    config = {k: v for k, v in vars(args).items() if k != "api_key_file"}
    config.update({"source_sha256": source_hash.hexdigest(), "python": sys.version,
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
