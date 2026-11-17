"""Check structured claims against tool facts; never certify prose as true."""
from __future__ import annotations

import math

FACT_FIELDS = {"train_accuracy", "search_accuracy", "generalization_gap", "training_seconds", "status"}


def fact_records(history):
    return [{"evidence_id": f"trial:{row['trial']}", "candidate": row["candidate"],
             **{k: row[k] for k in FACT_FIELDS if k in row}} for row in history]


def validate_scientific_decision(answer, history):
    for field in ("hypothesis", "alternative_explanation", "changed_factor"):
        if not isinstance(answer.get(field), str) or not answer[field].strip():
            raise ValueError(f"Missing scientific decision field: {field}")
    prediction = answer.get("prediction")
    if prediction != {"metric": "search_accuracy", "direction": "increase"}:
        raise ValueError("Prediction must concern search_accuracy increasing, not gap shrinking.")
    observations = answer.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observations must be a list of exact tool facts.")
    facts = {r["evidence_id"]: r for r in fact_records(history)}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Each observation must reference a tool fact.")
        source = facts.get(observation.get("evidence_id"))
        field = observation.get("field")
        if source is None or field not in FACT_FIELDS or field not in source:
            raise ValueError("Unknown evidence reference or metric.")
        actual, claimed = source[field], observation.get("value")
        if isinstance(actual, (int, float)):
            matches = type(claimed) in (int, float) and math.isfinite(claimed) and math.isclose(actual, claimed, rel_tol=0, abs_tol=1e-9)
        else:
            matches = claimed == actual
        if not matches:
            raise ValueError("Claimed observation does not match the recorded tool value.")
    completed = {r["candidate"]: r for r in history if r["status"] == "completed"}
    parent = answer.get("parent_candidate")
    if completed:
        if parent not in completed or not observations:
            raise ValueError("A follow-up hypothesis needs a completed parent and observed evidence.")
    elif parent is not None:
        raise ValueError("An initial hypothesis cannot reference an untested parent.")
    return answer


def hypothesis_outcome(decision, result, history):
    if result["status"] != "completed":
        return {"state": "execution_failed", "scope": "inner_search_only"}
    parent = next((r for r in history if r["candidate"] == decision.get("parent_candidate")
                   and r["status"] == "completed"), None)
    if parent is None or "prediction" not in decision:
        return {"state": "unassessed", "scope": "inner_search_only"}
    delta = result["search_accuracy"] - parent["search_accuracy"]
    state = "provisional_support" if delta >= .005 else "not_supported" if delta <= -.005 else "inconclusive"
    return {"state": state, "search_delta": delta, "scope": "inner_search_only",
            "causal_claim": False, "note": "Direction check only; no causal or statistical significance claim."}
