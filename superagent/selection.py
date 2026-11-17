"""Freeze search results, then optionally make one sealed confirmation choice."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from .models import predict_with_model


def frozen_shortlist(history, limit=2):
    completed = [row for row in history if row.get("status") == "completed"]
    return sorted(completed, key=lambda row: (-row["search_accuracy"], row["candidate"]))[:limit]


def confirm_once(kind, history, run_directory, x_confirmation, y_confirmation,
                 enabled, switch_threshold=.005, models_by_trial=None):
    """Choose only among the top two already-trained inner-search candidates.

    Confirmation has no callback into search; it cannot add a candidate or
    trigger retraining. Its score participates in selection and is logged as
    such rather than treated as an unbiased final metric.
    """
    shortlist = frozen_shortlist(history)
    record = {"version": 1, "enabled": enabled, "scope": "sealed_confirmation_only",
              "selection_threshold": switch_threshold,
              "shortlist": [{"candidate": row["candidate"], "trial": row["trial"],
                             "search_accuracy": row["search_accuracy"]} for row in shortlist],
              "refit": False}
    if not shortlist:
        record.update({"selected_candidate": None, "reason": "No completed search trial."})
        return None, None, record
    first = shortlist[0]
    if not enabled or len(shortlist) == 1:
        record.update({"selected_candidate": first["candidate"], "selected_trial": first["trial"],
                       "reason": "Confirmation disabled or fewer than two completed candidates.",
                       "switched": False})
        # No sealed evaluation takes place in this branch.  The runtime already
        # owns the selected trained model, so do not turn ordinary selection
        # into an unnecessary disk-reload requirement.  This also keeps the
        # normal execution path distinct from sealed confirmation below.
        if models_by_trial is not None and first["trial"] in models_by_trial:
            return first, models_by_trial[first["trial"]], record
        artifact = joblib.load(Path(run_directory) / f"trial_{first['trial']}.joblib")
        return first, artifact["model"], record
    scores, artifacts = {}, {}
    for row in shortlist:
        artifact = joblib.load(Path(run_directory) / f"trial_{row['trial']}.joblib")
        prediction = predict_with_model(kind, artifact["model"], x_confirmation)
        scores[row["candidate"]] = float(accuracy_score(y_confirmation, prediction))
        artifacts[row["candidate"]] = artifact
    second = shortlist[1]
    delta = scores[second["candidate"]] - scores[first["candidate"]]
    selected = second if delta >= switch_threshold else first
    record.update({"confirmation_rows": int(len(y_confirmation)), "scores": scores,
                   "comparison": {"challenger": second["candidate"], "incumbent": first["candidate"],
                                  "challenger_minus_incumbent": delta},
                   "selected_candidate": selected["candidate"], "selected_trial": selected["trial"],
                   "switched": selected["candidate"] != first["candidate"],
                   "reason": "One-time confirmation rule; no return to search."})
    return selected, artifacts[selected["candidate"]]["model"], record
