"""Frozen-model prediction stage.  It intentionally has no test evaluator."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import load_prediction_features
from .models import predict_with_model
from .acceptance import load_accepted_model, check_predictions


def create_predictions(task_package: str, run_dir: str | Path, output_path: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    if (run_path / "artifact_manifest.json").exists():
        manifest, model = load_accepted_model(run_path)
        features = load_prediction_features(task_package, manifest["kind"])
        predictions = check_predictions(predict_with_model(manifest["kind"], model, features),
                                        len(features), manifest["labels"])
        return _write_predictions(task_package, output_path, predictions, manifest["candidate"],
                                  model_sha256=manifest["model_sha256"])
    # Historical exports use the legacy format and are explicitly marked.
    selection = json.loads((run_path / "final_selection.json").read_text(encoding="utf-8"))
    specification = json.loads((run_path / "task_spec.json").read_text(encoding="utf-8"))
    if selection.get("test_accessed"):
        raise RuntimeError("This run directory is already marked as having accessed test data.")
    kind = specification["task_kind"]
    kind_to_loader = {
        "categorical_tabular_binary": "categorical_tabular",
        "pair_text_binary": "pair_text",
    }
    if kind not in kind_to_loader:
        raise ValueError(f"Unsupported frozen task kind: {kind}")
    model = joblib.load(run_path / selection["model_file"])
    features = load_prediction_features(task_package, kind_to_loader[kind])
    predictions = predict_with_model(kind_to_loader[kind], model, features)
    return _write_predictions(task_package, output_path, predictions, selection["selected_experiment"],
                              legacy_unverified=True)


def _write_predictions(task_package, output_path, predictions, selected_experiment, **extra):
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "prediction"])
        writer.writerows(enumerate(predictions.tolist()))
    manifest = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "selected_experiment": selected_experiment,
        "prediction_count": int(len(predictions)),
        "input_source": str(Path(task_package).resolve()),
        "test_labels_read": False,
        "prediction_file": str(destination),
        **extra,
    }
    (destination.parent / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
