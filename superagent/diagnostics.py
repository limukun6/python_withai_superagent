"""Deterministic, aggregate diagnostics for evidence-guided experiment choice.

The diagnostic payload intentionally contains no raw text or row-level labels.
All slice definitions are fitted on the inner fit partition before search.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def _missing_count(rows):
    values = np.asarray(rows, dtype=object).astype(str)
    missing = np.isin(np.char.lower(np.char.strip(values)), ["nan", "", "none", "null", "na", "?"])
    return missing.sum(axis=1)


def slice_definition(kind, x_fit):
    if kind == "pair_text":
        lengths = np.asarray([max(len(a.split()), len(b.split())) for a, b in x_fit])
        threshold = int(np.median(lengths))
        return {"kind": "max_pair_word_count", "threshold": threshold,
                "slices": [{"id": "short_or_equal", "rule": f"length <= {threshold}"},
                           {"id": "long", "rule": f"length > {threshold}"}]}
    counts = _missing_count(x_fit)
    return {"kind": "missing_feature_count", "threshold": 0,
            "slices": [{"id": "no_missing", "rule": "missing_count == 0"},
                       {"id": "has_missing", "rule": "missing_count > 0"}]}


def _slice_masks(kind, x_search, definition):
    if kind == "pair_text":
        values = np.asarray([max(len(a.split()), len(b.split())) for a, b in x_search])
        return {"short_or_equal": values <= definition["threshold"], "long": values > definition["threshold"]}
    values = _missing_count(x_search)
    return {"no_missing": values == 0, "has_missing": values > 0}


def _slice_metrics(y, prediction, masks, min_rows):
    rows = []
    for name, mask in masks.items():
        count = int(mask.sum())
        if count < min_rows:
            rows.append({"slice_id": name, "available": False, "count": count,
                         "reason": f"fewer than {min_rows} rows"})
            continue
        accuracy = float(np.mean(prediction[mask] == y[mask]))
        rows.append({"slice_id": name, "available": True, "count": count,
                     "accuracy": accuracy, "error_rate": 1.0 - accuracy})
    return rows


def summarize(kind, x_fit, x_search, y_search, history, run_directory, min_rows=50):
    """Return verifiable aggregate facts from completed inner-search trials."""
    definition = slice_definition(kind, x_fit)
    masks = _slice_masks(kind, x_search, definition)
    directory = Path(run_directory)
    reports, stored_predictions = [], {}
    for row in history:
        if row.get("status") != "completed":
            continue
        artifact_path = directory / f"trial_{row['trial']}.joblib"
        if not artifact_path.exists():
            continue
        artifact = joblib.load(artifact_path)
        prediction = np.asarray(artifact.get("search_predictions"))
        if prediction.shape != np.asarray(y_search).shape:
            reports.append({"evidence_id": f"diagnostic:trial:{row['trial']}",
                            "candidate": row["candidate"], "available": False,
                            "reason": "prediction shape mismatch"})
            continue
        stored_predictions[row["candidate"]] = prediction
        reports.append({"evidence_id": f"diagnostic:trial:{row['trial']}",
                        "candidate": row["candidate"], "available": True,
                        "slices": _slice_metrics(np.asarray(y_search), prediction, masks, min_rows)})
    pairwise = []
    candidates = sorted(stored_predictions)
    y_search = np.asarray(y_search)
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            a, b = stored_predictions[left], stored_predictions[right]
            pairwise.append({"left": left, "right": right,
                             "disagreement_rate": float(np.mean(a != b)),
                             "left_correct_right_wrong": int(np.sum((a == y_search) & (b != y_search))),
                             "right_correct_left_wrong": int(np.sum((b == y_search) & (a != y_search)))})
    return {"version": 1, "scope": "inner_search_only", "definition": definition,
            "min_rows": min_rows, "trial_reports": reports, "pairwise_predictions": pairwise}


def claim_facts(report):
    """Flatten aggregate output into exact, referenceable values for an LLM."""
    facts = []
    for trial in report.get("trial_reports", []):
        if not trial.get("available"):
            continue
        for row in trial["slices"]:
            if not row.get("available"):
                continue
            for metric in ("accuracy", "error_rate"):
                facts.append({"evidence_id": f"{trial['evidence_id']}:{row['slice_id']}",
                              "field": metric, "value": row[metric],
                              "candidate": trial["candidate"]})
    for pair in report.get("pairwise_predictions", []):
        facts.append({"evidence_id": f"diagnostic:pair:{pair['left']}:{pair['right']}",
                      "field": "disagreement_rate", "value": pair["disagreement_rate"],
                      "candidate": None})
    return facts
