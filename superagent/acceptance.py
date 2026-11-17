"""Independent acceptance of locally produced, trusted model artifacts.

joblib is not a safe format for models from untrusted third parties.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

from .models import predict_with_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_file") != "selected_model.joblib":
        raise ValueError("Unsupported model artifact manifest.")
    if digest(directory / manifest["model_file"]) != manifest["model_sha256"]:
        raise ValueError("Model file hash mismatch.")
    if digest(directory / "task_contract.json") != manifest["contract_sha256"]:
        raise ValueError("Task contract hash mismatch.")
    contract = json.loads((directory / "task_contract.json").read_text(encoding="utf-8"))
    if manifest["kind"] != contract["kind"] or manifest["labels"] != contract["labels"]:
        raise ValueError("Artifact and task contract disagree.")
    return manifest


def check_predictions(predictions, count, labels):
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or len(predictions) != count:
        raise ValueError("Prediction shape or row count mismatch.")
    if not set(predictions.tolist()).issubset(set(labels)):
        raise ValueError("Prediction contains undeclared labels.")
    return predictions


def load_accepted_model(directory):
    directory = Path(directory)
    manifest = load_manifest(directory)
    acceptance = json.loads((directory / "acceptance.json").read_text(encoding="utf-8"))
    if not acceptance.get("passed") or acceptance.get("model_sha256") != manifest["model_sha256"]:
        raise ValueError("Model has not passed independent acceptance.")
    if acceptance.get("manifest_sha256") != digest(directory / "artifact_manifest.json"):
        raise ValueError("Manifest changed after acceptance.")
    bundle = joblib.load(directory / manifest["model_file"])
    if bundle["kind"] != manifest["kind"] or bundle["candidate"] != manifest["candidate"]:
        raise ValueError("Model bundle disagrees with manifest.")
    return manifest, bundle["model"]


def accept_artifact(directory, features, expected_predictions, timeout=60):
    directory = Path(directory).resolve()
    joblib.dump({"features": features, "expected_predictions": expected_predictions},
                directory / "acceptance_probe.joblib")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "superagent.acceptance", str(directory)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout)
        report = json.loads((directory / "acceptance.json").read_text(encoding="utf-8"))
        if process.returncode and report.get("passed"):
            raise ValueError("Acceptance process returned an error despite a pass report.")
        return report
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        (directory / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main():
    directory = Path(sys.argv[1])
    try:
        manifest = load_manifest(directory)
        bundle = joblib.load(directory / manifest["model_file"])
        if bundle["kind"] != manifest["kind"] or bundle["candidate"] != manifest["candidate"]:
            raise ValueError("Model bundle does not match manifest.")
        probe = joblib.load(directory / "acceptance_probe.joblib")
        features, expected = probe["features"], np.asarray(probe["expected_predictions"])
        if not len(features):
            raise ValueError("Acceptance needs at least one inner-search sample.")
        predictions = check_predictions(predict_with_model(bundle["kind"], bundle["model"], features),
                                        len(features), manifest["labels"])
        if not np.array_equal(predictions, expected):
            raise ValueError("Reloaded model predictions differ from the frozen model.")
        for index in range(min(8, len(features))):
            single = predict_with_model(bundle["kind"], bundle["model"], features[index:index+1])
            if not np.array_equal(np.asarray(single), predictions[index:index+1]):
                raise ValueError("Single-row and batch predictions disagree.")
        reversed_features = features[::-1]
        if not np.array_equal(predict_with_model(bundle["kind"], bundle["model"], reversed_features), predictions[::-1]):
            raise ValueError("Prediction order is not preserved.")
        report = {"passed": True, "model_sha256": manifest["model_sha256"],
                  "manifest_sha256": digest(directory / "artifact_manifest.json"),
                  "probe_rows": len(features), "scope": "inner_search_inputs_only",
                  "checks": ["hashes", "reload_equivalence", "labels", "row_count", "batch_single", "row_order"]}
    except Exception as error:
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
