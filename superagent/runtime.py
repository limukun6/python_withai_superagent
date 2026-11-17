"""Single task execution core shared by CLI and benchmark."""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
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
from .contracts import build_contract, describe_candidate
from .decisions import fact_records, validate_scientific_decision, hypothesis_outcome
from .acceptance import accept_artifact, digest, check_predictions
from .diagnostics import summarize as summarize_diagnostics, claim_facts as diagnostic_claim_facts
from .selection import confirm_once

STRATEGIES = ("random", "deterministic", "oneshot", "feedback", "reviewed", "scientist")
RUNTIME_VERSION = "foundation-1"


def take(rows, indices):
    return rows[indices] if isinstance(rows, np.ndarray) else [rows[int(i)] for i in indices]


def _groups(task):
    # Identical rows (or reversed text pairs) stay together across the inner split.
    keys = [tuple(sorted(row)) if task.kind == "pair_text" else tuple(map(str, row)) for row in task.X_train]
    return np.asarray([hashlib.sha256(repr(key).encode()).hexdigest() for key in keys])


def inner_split(task, seed):
    groups = _groups(task)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, search = next(splitter.split(np.zeros(len(task.y_train)), task.y_train, groups))
    return train, search


def confirmation_split(task, seed):
    """Make disjoint fit/search/confirmation partitions from official train."""
    groups = _groups(task)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(task.y_train)), task.y_train, groups))
    confirmation = folds[0][1]
    search = folds[1][1]
    fit = np.asarray(sorted(set(range(len(task.y_train))) - set(confirmation) - set(search)))
    if not len(fit) or set(fit) & set(search) or set(fit) & set(confirmation) or set(search) & set(confirmation):
        raise ValueError("Unable to construct disjoint fit/search/confirmation partitions.")
    return fit, search, confirmation


class Policy:
    def __init__(self, strategy, seed, client, logger, limit):
        self.strategy, self.client, self.logger, self.limit = strategy, client, logger, limit
        self.rng = random.Random(seed)
        self.plan = None
        self.last_decision = {}

    def choose(self, data_profile, history, available, diagnostics=None):
        ids = [item["id"] for item in available]
        if self.strategy in ("random", "deterministic"):
            chosen = self.rng.choice(ids) if self.strategy == "random" else ids[0]
            reason = "Uniform random draw without replacement." if self.strategy == "random" else "Next candidate in the fixed registry order."
            self.last_decision = {"candidate": chosen, "reason": reason, "mode": self.strategy}
            return chosen, reason
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
        elif self.strategy == "scientist":
            diagnostics = diagnostics or {}
            diagnostic_facts = diagnostic_claim_facts(diagnostics)
            answer = self.client.ask({**context, "tool_facts": fact_records(history) + diagnostic_facts,
                "diagnostics": diagnostics,
                "instruction": (
                    'Return one JSON object: {"candidate": exact_available_id, "reason": string, '
                    '"observations": [{"evidence_id": "trial:N", "field": exact_tool_field, "value": exact_tool_value}], '
                    '"hypothesis": string, "alternative_explanation": string, "changed_factor": string, '
                    '"parent_candidate": completed_candidate_id_or_null, '
                    '"prediction": {"metric": "search_accuracy", "direction": "increase"}}. '
                    'Facts must exactly match tool_facts. With completed trials cite at least one fact and a completed parent. '
                    'Initially use empty observations and null parent. A smaller train/search gap is not itself success. '
                    'Describe multiple changed factors honestly; hypotheses are not established facts. '
                    'Use the actual task objective and estimator metadata; do not assume text pairs mean similarity.')},
                self.logger, "scientist")
            validate_scientific_decision(answer, history, diagnostic_facts)
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
        self.last_decision = dict(answer)
        return answer["candidate"], answer["reason"]


def train_candidate(payload_path, candidate, destination, timeout):
    started = time.monotonic()
    command = [sys.executable, "-m", "superagent.experiment_worker", str(payload_path), candidate, str(destination)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                cwd=Path(__file__).resolve().parents[1])
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout", "error": "Training time limit reached", "training_seconds": time.monotonic() - started}
    if result.returncode:
        return None, {"status": "failed", "error": result.stderr[-3000:], "training_seconds": time.monotonic() - started}
    artifact = joblib.load(destination)
    return artifact["model"], {"status": "completed", **artifact["metrics"]}


def run_task(task, strategy, seed, directory, client, max_trials=3, time_limit=600, trial_timeout=120,
             confirmation=False):
    if strategy not in STRATEGIES or not 1 <= max_trials <= 14:
        raise ValueError("Unknown strategy or trial limit outside 1..14.")
    if any(not math.isfinite(t) or t <= 0 for t in (time_limit, trial_timeout)):
        raise ValueError("Time limits must be finite and positive.")
    if strategy not in ("random", "deterministic") and client is None:
        raise ValueError("This strategy requires a configured LLM client.")
    contract = build_contract(task)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(directory)
    logger.initialize_usage_ledger()
    logger.write_json("task_contract.json", contract.to_dict())
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    logger.write_json("runtime_manifest.json", {"runtime_version": RUNTIME_VERSION,
                      "source_sha256": source_hash.hexdigest(), "python": sys.version,
                      "strategy": strategy, "seed": seed, "max_trials": max_trials,
                      "search_time_limit": time_limit, "trial_timeout": trial_timeout,
                      "split_protocol": "grouped_fit_search_confirmation" if confirmation else "grouped_inner_80_20_no_confirmation",
                      "test_accessed": False})
    logger.event("state", state="SEARCH")
    start = time.monotonic()
    if confirmation:
        train_idx, search_idx, confirmation_idx = confirmation_split(task, seed)
    else:
        train_idx, search_idx = inner_split(task, seed)
        confirmation_idx = None
    split_record = {"seed": seed, "train_indices": train_idx.tolist(), "search_indices": search_idx.tolist()}
    if confirmation_idx is not None:
        split_record["confirmation_indices"] = confirmation_idx.tolist()
    logger.write_json("split.json", split_record)
    payload = {"kind": task.kind, "X_train": take(task.X_train, train_idx), "y_train": task.y_train[train_idx],
               "X_search": take(task.X_train, search_idx), "y_search": task.y_train[search_idx],
               "search_indices": search_idx.tolist()}
    payload_path = directory / "inner_data.joblib"
    joblib.dump(payload, payload_path)
    inner = DevelopmentTask(task.kind, "inner_split", payload["X_train"], payload["y_train"],
                            payload["X_search"], payload["y_search"], {})
    data_profile = profile(inner)
    data_profile["task_contract"] = contract.to_dict()
    logger.write_json("data_profile.json", data_profile)
    available = [describe_candidate(task.kind, c) for c in search_candidates(task.kind)]
    logger.write_json("candidate_registry.json", available)
    policy = Policy(strategy, seed, client, logger, max_trials)
    history, best_model, best_id, best_score = [], None, None, -1.0
    models_by_trial = {}
    fallbacks, blocked_llm = 0, False
    random_fallback = random.Random(seed)
    for index in range(max_trials):
        if time.monotonic() - start >= time_limit:
            break
        try:
            if blocked_llm:
                raise RuntimeError("Planner disabled after earlier failure.")
            diagnostics = summarize_diagnostics(task.kind, payload["X_train"], payload["X_search"],
                                                payload["y_search"], history, directory)
            logger.write_json("diagnostics.json", diagnostics)
            if client is not None:
                client.deadline = start + time_limit
            chosen, reason = policy.choose(data_profile, history, available, diagnostics=diagnostics)
            decision = policy.last_decision
        except Exception as error:
            blocked_llm = True
            fallbacks += 1
            chosen = random_fallback.choice(available)["id"]
            reason = "Random recovery after planner failure; this is not a valid pure LLM run."
            logger.event("planner_fallback", error=str(error), chosen=chosen)
            decision = {"candidate": chosen, "reason": reason, "mode": "random_recovery"}
        remaining_seconds = time_limit - (time.monotonic() - start)
        if remaining_seconds <= 0:
            break
        configuration = next(item for item in available if item["id"] == chosen)
        available = [item for item in available if item["id"] != chosen]
        logger.event("decision", trial=index + 1, candidate=chosen, reason=reason, params=configuration["params"])
        logger.event("decision_record", trial=index + 1, decision=decision,
                     evidence_checks="passed" if strategy == "scientist" and not blocked_llm else "not_applicable",
                     interpretation="Hypothesis, not verified causal explanation.")
        model, metrics = train_candidate(payload_path, chosen, directory / f"trial_{index+1}.joblib",
                                         min(trial_timeout, remaining_seconds))
        row = {"trial": index + 1, "candidate": chosen, "params": configuration["params"],
               "allowed_training_seconds": min(trial_timeout, remaining_seconds), **metrics}
        if model is not None:
            models_by_trial[index + 1] = model
        if model is not None and (metrics["search_accuracy"] > best_score or
            (metrics["search_accuracy"] == best_score and chosen < best_id)):
            best_score, best_id, best_model = metrics["search_accuracy"], chosen, model
        row["best_search_accuracy"] = None if best_model is None else best_score
        row["elapsed_seconds"] = time.monotonic() - start
        outcome = hypothesis_outcome(decision, row, history)
        logger.event("hypothesis_outcome", trial=index + 1, **outcome)
        history.append(row)
        logger.event("trial_result", **row)
        logger.write_json("trials.json", history)
        # Refresh after every trial so the final artifact also includes the
        # last experiment; the next planner turn consumes the same definition.
        logger.write_json("diagnostics.json", summarize_diagnostics(
            task.kind, payload["X_train"], payload["X_search"], payload["y_search"], history, directory))
    # Freeze before sealed confirmation/official validation. No refit.
    selected_row, selected_model, confirmation_record = confirm_once(
        task.kind, history, directory,
        take(task.X_train, confirmation_idx) if confirmation_idx is not None else None,
        task.y_train[confirmation_idx] if confirmation_idx is not None else None,
        confirmation, models_by_trial=models_by_trial)
    logger.write_json("confirmation.json", confirmation_record)
    best_id = None if selected_row is None else selected_row["candidate"]
    best_score = None if selected_row is None else selected_row["search_accuracy"]
    best_model = selected_model
    logger.write_json("selection.json", {"candidate": best_id, "search_accuracy": best_score,
                      "selection_rule": "sealed top-two confirmation" if confirmation else "max inner accuracy; ties by candidate id",
                      "refit": False, "confirmation_enabled": confirmation,
                      "confirmation_switched": confirmation_record.get("switched", False)})
    logger.event("state", state="FROZEN")
    official_accuracy = None
    acceptance = {"passed": False, "error": "No candidate completed."}
    if best_model is not None:
        joblib.dump({"kind": task.kind, "candidate": best_id, "model": best_model}, directory / "selected_model.joblib")
        logger.write_json("artifact_manifest.json", {"schema_version": 1, "runtime_version": RUNTIME_VERSION,
                          "kind": task.kind, "candidate": best_id, "labels": contract.labels,
                          "model_file": "selected_model.joblib", "model_sha256": digest(directory / "selected_model.joblib"),
                          "contract_sha256": digest(directory / "task_contract.json"), "test_accessed": False})
        logger.event("state", state="ACCEPTANCE")
        probe = take(payload["X_search"], np.arange(min(32, len(payload["y_search"]))))
        try:
            expected = predict_with_model(task.kind, best_model, probe)
            acceptance = accept_artifact(directory, probe, expected)
        except Exception as error:
            acceptance = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            logger.write_json("acceptance.json", acceptance)
        if acceptance["passed"]:
            logger.event("state", state="EXTERNAL_VALIDATION")
            predictions = check_predictions(predict_with_model(task.kind, best_model, task.X_val), len(task.y_val), contract.labels)
            official_accuracy = float(accuracy_score(task.y_val, predictions))
            logger.write_json("validation_predictions.json", predictions.tolist())
    else:
        logger.write_json("acceptance.json", acceptance)
    usage = [json.loads(line) for line in logger.usage_path.read_text(encoding="utf-8").splitlines()]
    summary = {"strategy": strategy, "seed": seed, "task_kind": task.kind,
        "selected_candidate": best_id, "search_accuracy": best_score if best_model else None,
        "official_validation_accuracy": official_accuracy,
        "completed": acceptance["passed"], "valid_policy_run": acceptance["passed"] and fallbacks == 0,
        "acceptance_passed": acceptance["passed"], "runtime_version": RUNTIME_VERSION,
        "confirmation_enabled": confirmation, "confirmation_switched": confirmation_record.get("switched", False),
        "fallback_count": fallbacks, "trials_attempted": len(history),
        "failed_trials": sum(r["status"] != "completed" for r in history),
        "elapsed_seconds": time.monotonic() - start,
        "llm_calls": sum(u["event"] != "ledger_initialized" for u in usage),
        "tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage),
        "estimated_cost_rmb": sum(u.get("cost_rmb") or 0 for u in usage),
        "usage_complete": not any(u.get("usage_known") is False for u in usage),
        "test_accessed": False, "run_directory": str(directory)}
    logger.write_json("summary.json", summary)
    logger.event("state", state="COMPLETED" if summary["completed"] else "FAILED")
    return summary
