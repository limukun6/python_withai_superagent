"""Train, reload and verify predictions on anonymous multiclass fixtures."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from superagent.adaptive_local import save, sha


def archive_array(features, labels=None):
    stream = io.BytesIO()
    arrays = {"X": features}
    if labels is not None:
        arrays["y"] = labels
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def make_package(path, modality, validation):
    rng = np.random.default_rng(741)
    classes = np.array(["red", "green", "blue"])
    class_ids = np.arange(120) % len(classes)
    if modality == "image":
        features = rng.integers(0, 80, size=(120, 12, 12, 3), dtype=np.uint8)
        for i, class_id in enumerate(class_ids):
            features[i, 2:8, 3:7, class_id] += 160
    else:
        features = rng.normal(size=(120, 6))
        features[np.arange(120), class_ids] += 4
    batches = {"train": (features[:90], classes[class_ids[:90]]), "test": (features[105:], None)}
    if validation:
        batches["validation"] = (features[90:105], classes[class_ids[90:105]])
    manifest = dict(schema_version=1, modality=modality, metric="macro_f1",
                    splits={key: {"format": "npz", "path": f"{key}.npz"} for key in batches})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        out.writestr("manifest.json", json.dumps(manifest))
        for split, (x, y) in batches.items():
            out.writestr(f"{split}.npz", archive_array(x, y))
    return set(classes.tolist()), len(batches["test"][0])


def command(log, *args, success=True):
    env = os.environ | dict(SUPERAGENT_OFFLINE="1", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-m", "superagent", *map(str, args)],
                            env=env, capture_output=True, encoding="utf-8", timeout=660)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if (result.returncode == 0) != success:
        raise RuntimeError(f"Unexpected return code {result.returncode}; see {log}")
    return result.stderr


def check_rows(rows, classes, count):
    if [row["row_id"] for row in rows] != [str(i) for i in range(count)]:
        raise ValueError("Prediction rows are missing, duplicated or reordered")
    if not {row["prediction"] for row in rows} <= classes:
        raise ValueError("Prediction contains an undeclared label")


def case(root, name, modality, validation, trials):
    folder = root / name
    folder.mkdir()
    package = folder / "classification.zip"
    classes, count = make_package(package, modality, validation)
    run = folder / "run"
    command(folder / "run.log", "run", package, "--run-dir", run, "--planner", "local",
            "--policy", "interleaved_routes", "--portfolio", "expanded", "--trials", trials,
            "--trial-timeout", "90", "--time-limit", "600", "--feature-cache", "run")
    outcome = json.loads((run / "run_outcome.json").read_text(encoding="utf-8"))
    if (not outcome["requested_planner_completed"] or outcome["primary_metric"] != "macro_f1"
            or outcome["provided_validation_available"] != validation
            or outcome["new_api_requests"] != 0 or outcome["formal_test_accessed"]):
        raise ValueError(f"Unexpected unified-run outcome for {name}")
    predict = folder / "predictions.csv"
    command(folder / "predict.log", "predict", package, "--run-dir", run, "--output", predict)
    with predict.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    check_rows(rows, classes, count)
    try:
        check_rows(rows[1:] + rows[:1], classes, count)
    except ValueError:
        pass
    else:
        raise ValueError("Row-order negative control was not detected")
    model_dir = folder / "altered_bundle"
    model_dir.mkdir()
    for filename in ("task_spec.json", "final_selection.json", "selected_model.joblib"):
        shutil.copyfile(run / filename, model_dir / filename)
    with (model_dir / "selected_model.joblib").open("ab") as altered:
        altered.write(b"changed")
    error = command(folder / "altered.log", "predict", package, "--run-dir", model_dir,
                    "--output", folder / "unexpected.csv", success=False)
    if "no longer matches the frozen selection hash" not in error:
        raise ValueError(f"Unexpected reason for rejecting altered bundle: {name}")
    prediction_manifest = json.loads((folder / "prediction_manifest.json").read_text(encoding="utf-8"))
    if prediction_manifest["test_labels_read"] or prediction_manifest["prediction_count"] != count:
        raise ValueError(f"Unexpected prediction manifest: {name}")
    return dict(modality=modality, primary_metric=outcome["primary_metric"], validation=validation,
                validation_score=outcome["validation_score"], predicted_rows=count,
                full_refit=outcome["full_refit_completed"], row_reordering_detected=True,
                altered_bundle_rejected=True, prediction_sha256=sha(predict),
                fitted_candidate=outcome["selected_candidate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, modality, validation in (("fixture_a", "tabular", True), ("fixture_b", "image", False)):
        results[name] = case(root, name, modality, validation, args.trials)
        save(root / "results.json", results)
        print(json.dumps({name: results[name]}), flush=True)


if __name__ == "__main__":
    main()
