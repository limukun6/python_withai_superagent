"""Course entry point for autonomous classification on a complete task package."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from superagent.cli import build_parser
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.predict import create_predictions
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups


# The independent audit uses the project's original grouped-development protocol.
AUDIT_SEED = 41101


def audit_partition(task):
    schema = task.spec.get("input_schema", {})
    if (task.kind != "pair_text" or schema.get("pair_semantics") != "preference"
            or schema.get("pair_label_one") != "left"):
        return None, "no declared directional preference"
    groups = sample_groups(task, "shared_text")
    if len(np.unique(groups)) == len(groups):
        return None, "no repeated text components"
    if any(len(np.unique(groups[task.y_train == label])) < 10 for label in np.unique(task.y_train)):
        return None, "insufficient independent components for an outer and an inner five-fold split"
    try:
        train, reserve, info = grouped_partition(task, AUDIT_SEED, "shared_text")
    except ValueError as exc:
        if str(exc).startswith("No leakage-free grouped partition"):
            return None, "no class-preserving shared-text partition"
        raise
    if info["folds"] < 5:
        return None, "a five-fold outer audit is unavailable"
    return (train, reserve, info), "repeated text components with a class-preserving audit"


def pair_packet(path, task, indexes, *, training):
    schema = task.spec["input_schema"]
    columns = [column["name"] for column in schema["columns"]]
    split = "train" if training else "test"
    manifest = {
        "schema_version": 1, "objective": "classification", **schema,
        "metric": task.spec.get("metric", "accuracy"),
        "splits": {split: {"format": "csv", "path": "rows.csv"}},
    }
    for name in ("positive_label", "evaluation_protocol"):
        if name in task.spec:
            manifest[name] = task.spec[name]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([*columns, schema["target"]] if training else columns)
    for index in indexes:
        pair = task.X_train[index]
        writer.writerow([*pair, task.y_train[index]] if training else pair)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("rows.csv", stream.getvalue())


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def train(package, run_dir):
    task = load_training_task(package)
    partition, reason = audit_partition(task)
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = MetricSpec.for_task(task)
    record = {"entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_training_rows": len(task.y_train), "partition_reason": reason,
              "model_training_rows": len(task.y_train), "audit_rows": 0}
    training_path = package
    if partition is not None:
        fit, held, info = partition
        training_path = run_dir / "training.zip"
        pair_packet(training_path, task, fit, training=True)
        audit_path = run_dir / "audit_features.zip"
        pair_packet(audit_path, task, held, training=False)
        truth = task.y_train[held].copy()
        record.update(model_training_rows=len(fit), audit_rows=len(held), audit_seed=AUDIT_SEED,
                      fit_row_ids_sha256=fingerprint(fit.tolist()), audit_row_ids_sha256=fingerprint(held.tolist()),
                      partition=info)
    del task
    save(run_dir / "training_protocol.json", record)
    native = build_parser().parse_args(["unified", str(training_path), "--run-dir", str(run_dir / "agent")])
    code = native.handler(native)
    if code or partition is None:
        return code
    output = create_predictions(str(audit_path), run_dir / "agent", run_dir / "audit" / "predictions.csv")
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["row_id"] for row in rows] != list(map(str, range(len(truth)))):
        raise ValueError("Audit predictions and held-out rows disagree")
    predictions = np.asarray([row["prediction"] for row in rows], dtype=truth.dtype)
    probabilities = order = None
    if spec.requires_scores:
        receipt = json.loads((output.parent / "prediction_manifest.json").read_text(encoding="utf-8"))
        order = receipt["score_class_order"]
        probabilities = [[float(row[f"score_{i}"]) for i in range(len(order))] for row in rows]
    metrics = spec.evaluate(truth, predictions, probabilities, order)
    metrics.update(correct=int(np.sum(truth == predictions)), scope="training-only, after final selection",
                   model_sha256=json.loads((run_dir / "agent/final_selection.json").read_text())["model_sha256"],
                   prediction_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    save(run_dir / "audit.json", metrics)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "predict"):
        command = commands.add_parser(action)
        command.add_argument("task_package", type=Path)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--weights",type=Path,help="Root containing locally pinned encoders")
        command.add_argument("--feature-bank",type=Path,help="Optional cache of frozen input-only representations")
        if action == "predict":
            command.add_argument("--output", required=True, type=Path)
        else:
            command.add_argument("--strategy", choices=("default", "crossfit-challenger", "pilot-accuracy", "pilot-brier",
                                                        "pilot-no-backup", "pilot-backup"), default="default")
            command.add_argument("--fit-cache", type=Path, help="Optional trusted local pilot component cache")
    args = parser.parse_args()
    if args.feature_bank is not None and args.weights is None:
        parser.error("--feature-bank requires --weights")
    if args.action == "train":
        if args.strategy in ("pilot-accuracy", "pilot-brier", "pilot-no-backup", "pilot-backup"):
            from tools.pilot_challenger import train as train_pilot
            return train_pilot(args.task_package, args.run_dir, args.weights, args.feature_bank, args.fit_cache, args.strategy)
        if args.fit_cache is not None:
            parser.error("--fit-cache requires a pilot strategy")
        if args.strategy=="crossfit-challenger":
            from tools.challenger_training import train as train_challenger
            return train_challenger(args.task_package,args.run_dir,args.weights,args.feature_bank)
    if args.weights is not None:
        from tools.portfolio_training import configure
        configure(args.weights,args.feature_bank)
    if args.action == "train":
        return train(args.task_package, args.run_dir)
    path = create_predictions(str(args.task_package), args.run_dir / "agent", args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
