"""Isolated training worker; receives only inner training/search data."""
from __future__ import annotations

import sys
import time
import joblib
from sklearn.metrics import accuracy_score, recall_score
from .models import search_candidates, predict_with_model


def main() -> None:
    payload_path, candidate_id, output_path = sys.argv[1:]
    payload = joblib.load(payload_path)
    candidate = next(c for c in search_candidates(payload["kind"]) if c.family == candidate_id)
    start = time.perf_counter()
    model, predictions = candidate.fit_predict(payload["X_train"], payload["y_train"], payload["X_search"])
    training_predictions = predict_with_model(payload["kind"], model, payload["X_train"])
    train_accuracy = float(accuracy_score(payload["y_train"], training_predictions))
    search_accuracy = float(accuracy_score(payload["y_search"], predictions))
    joblib.dump({"model": model, "metrics": {
        "train_accuracy": train_accuracy,
        "search_accuracy": search_accuracy,
        "generalization_gap": train_accuracy - search_accuracy,
        "class_recall": recall_score(payload["y_search"], predictions, average=None, zero_division=0).tolist(),
        "training_seconds": time.perf_counter() - start,
    }}, output_path)


if __name__ == "__main__":
    main()
