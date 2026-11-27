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
from superagent.experiment_policy import identity
from superagent.exploration_rule import load_rule
from superagent.metrics import MetricSpec
from superagent.planner_review import add_review_arguments, feedback_mode, review_flags
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


def train_legacy(package, run_dir):
    from superagent.task_constraints import package_restrictions
    if package_restrictions(package):
        return train(package,run_dir)
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
    native = build_parser().parse_args(["unified", str(training_path), "--run-dir", str(run_dir / "agent"),
                                        "--policy", "legacy", "--fit-limit", "10"])
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


def train(package, run_dir, policy='autonomous', *, fit_limit=None, time_limit=None, self_review=False,
          exploration_rule=None, feedback_reflection=None, reflection_history=None,
          task_interpretation='fallback',prediction_input=None,semantic_artifact_from=None):
    if policy == 'legacy':
        if task_interpretation=='auto':task_interpretation='fallback'
        if (fit_limit is not None or time_limit is not None or self_review or exploration_rule is not None
                or feedback_reflection or reflection_history or task_interpretation!='fallback' or prediction_input is not None
                or semantic_artifact_from is not None):
            raise ValueError('Budget overrides apply to the autonomous course entry point')
        return train_legacy(package, run_dir)
    arguments=prepare_training(package,run_dir,fit_limit,time_limit,self_review,exploration_rule,feedback_reflection,reflection_history,
                               task_interpretation=task_interpretation,prediction_input=prediction_input,semantic_artifact_from=semantic_artifact_from)
    return arguments.handler(arguments)


def prepare_training(package,run_dir,fit_limit=None,time_limit=None,self_review=False,exploration_rule=None,
                     feedback_reflection=None,reflection_history=None,*,task_interpretation='fallback',prediction_input=None,semantic_artifact_from=None):
    from superagent.feedback_reflection import load_memory
    if reflection_history and not feedback_reflection:
        raise ValueError('A reflection history needs an enabled reviewer')
    prior = load_memory(reflection_history)
    rule = load_rule(exploration_rule) if exploration_rule is not None else None
    run_dir.mkdir(parents=True,exist_ok=False)
    flags=['unified',str(package),'--run-dir',str(run_dir/'agent'),'--task-interpretation',task_interpretation]
    if prediction_input is not None: flags.extend(['--prediction-input',str(Path(prediction_input).resolve())])
    if semantic_artifact_from is not None: flags.extend(['--semantic-artifact-from',str(Path(semantic_artifact_from).resolve())])
    extra = {}
    if feedback_reflection:
        flags.extend(['--feedback-reflection', feedback_reflection])
        extra['feedback_reflection'] = feedback_reflection
    if prior:
        history_path = run_dir.resolve()/'reflection_history.json'
        save(history_path,prior)
        flags.extend(['--reflection-history', str(history_path)])
        extra['reflection_history_digest'] = identity(prior)
    if rule is not None:
        frozen = run_dir.resolve()/'exploration_rule.json'
        save(frozen, rule)
        flags.extend(['--exploration-rule', str(frozen)])
        extra['exploration_rule_digest'] = identity(rule)
    flags.extend(review_flags(self_review))
    if fit_limit is not None:
        flags.extend(['--fit-limit',str(fit_limit)])
    if time_limit is not None:
        flags.extend(['--time-limit',str(time_limit)])
    arguments=build_parser().parse_args(flags)
    save(run_dir/'training_protocol.json',dict(policy='anonymous-sequential-v1',
         entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         selection_data='training-only groups',final_refit='all supplied training rows',time_limit=arguments.time_limit,
         fit_limit=arguments.fit_limit,self_review=self_review,task_interpretation=task_interpretation,
         prediction_input=str(prediction_input) if prediction_input is not None else None,
         semantic_artifact_from=str(semantic_artifact_from) if semantic_artifact_from is not None else None,**extra))
    return arguments


def solve_training(package, run_dir, *, fit_limit=None, time_limit=None, self_review=False, exploration_rule=None,
                   feedback_reflection=None, reflection_history=None, task_interpretation='fallback',
                   prediction_input=None, semantic_artifact_from=None, output=None):
    """Only intake may check unlabeled prediction structure. Export and training use separate processes."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from superagent.autonomous_checkpoint import interrupted
    package, run_dir = Path(package).resolve(), Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError('A fresh solve needs a new run directory. Existing frozen models support predict.')
    preview = output is not None
    if output is not None and prediction_input is None:
        raise ValueError('First delivery needs both prediction features and an output path')
    if preview:
        prediction_input, output = Path(prediction_input).resolve(), Path(output).resolve()
        from superagent.first_delivery import guard_output
        guard_output(output, run_dir, (package, prediction_input))
    arguments=prepare_training(package,run_dir,fit_limit,time_limit,self_review,exploration_rule,feedback_reflection,reflection_history,
                               task_interpretation=task_interpretation,prediction_input=prediction_input,semantic_artifact_from=semantic_artifact_from)
    inherited = ['--exploration-rule', arguments.exploration_rule] if arguments.exploration_rule else []
    if feedback_reflection:
        inherited.extend(['--feedback-reflection',feedback_reflection])
    if arguments.reflection_history:
        inherited.extend(['--reflection-history',arguments.reflection_history])
    if preview:
        inherited.append('--await-first-delivery')
    directory = Path(__file__).resolve().parent
    deadline = time.monotonic()+arguments.time_limit+30
    inherited.extend(['--task-interpretation',task_interpretation])
    if prediction_input is not None: inherited.extend(['--prediction-input',str(Path(prediction_input).resolve())])
    if semantic_artifact_from is not None: inherited.extend(['--semantic-artifact-from',str(Path(semantic_artifact_from).resolve())])
    child = subprocess.Popen([sys.executable,'-m','superagent','unified',str(package),'--run-dir',str(run_dir/'agent'),
                              '--fit-limit',str(arguments.fit_limit),'--time-limit',str(arguments.time_limit),
                              *review_flags(self_review), *inherited], cwd=directory,
                             stdin=subprocess.PIPE if preview else subprocess.DEVNULL, start_new_session=os.name=='posix')
    def stop_child(force):
        if child.poll() is None:
            if os.name == 'posix':
                try:
                    os.killpg(child.pid, signal.SIGKILL if force else signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif force:
                child.kill()
            else:
                child.terminate()
        child.wait()
    try:
        if preview:
            prior = run_dir/'agent'/'prior_selection.json'
            while not prior.is_file() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(.05)
            if prior.is_file() and child.poll() is None:
                # The exporter has no planner or training callback. Training receives the same byte even
                # if the prediction input is invalid; its choice of experiments cannot depend on the export.
                with (run_dir/'first_delivery.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','superagent.first_delivery',str(prediction_input),
                                    str(run_dir),str(output)], cwd=directory,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.monotonic()))
                try:
                    child.stdin.write(b'G')
                    child.stdin.flush()
                except BrokenPipeError:
                    pass
        code = child.wait(timeout=max(.01,deadline-time.monotonic()))
        reason = 'Training process exited with status '+str(code)
    except subprocess.TimeoutExpired:
        stop_child(True)
        code, reason = 2, 'Training exceeded its configured wall time plus 30-second shutdown grace'
    except KeyboardInterrupt:
        stop_child(False)
        raise
    finally:
        if child.stdin is not None:
            try:
                child.stdin.close()
            except BrokenPipeError:
                pass
    if code and interrupted(run_dir/'agent',reason):
        return 0
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "predict", "solve"):
        command = commands.add_parser(action)
        command.add_argument("task_package", type=Path)
        command.add_argument("--run-dir", required=True, type=Path)
        if action in ('train','solve'):
            command.add_argument('--fit-limit',type=int)
            command.add_argument('--time-limit',type=float)
            add_review_arguments(command)
            command.add_argument('--task-interpretation',choices=['auto','fallback','proactive'],default='auto',
                                 help='auto reviews documents when a model is configured; fallback keeps the local-first path')
            command.add_argument('--semantic-artifact-from',type=Path)
        if action in ("predict","solve"):
            command.add_argument("--output", required=True, type=Path)
        if action == "solve":
            command.add_argument("--prediction-input", required=True, type=Path,
                                 help="Unlabeled features for pre-fit schema checks and frozen-model prediction")
        if action == "train":
            command.add_argument('--policy',choices=['autonomous','legacy'],default='autonomous')
            command.add_argument('--prediction-input',type=Path,help='Optional unlabeled input for pre-fit checks')
    args = parser.parse_args()
    args.feedback_reflection=feedback_mode(args)
    if args.action == "train":
        return train(args.task_package, args.run_dir, args.policy,fit_limit=args.fit_limit,time_limit=args.time_limit,
                     self_review=args.self_review,exploration_rule=args.exploration_rule,
                     feedback_reflection=args.feedback_reflection,reflection_history=args.reflection_history,
                     task_interpretation=args.task_interpretation,prediction_input=args.prediction_input,semantic_artifact_from=args.semantic_artifact_from)
    source=args.task_package
    if args.action == 'predict':
        from superagent.first_delivery import guard_output
        guard_output(args.output.resolve(), args.run_dir.resolve(), (source.resolve(),))
    if args.action=='solve':
        code=solve_training(source,args.run_dir,fit_limit=args.fit_limit,time_limit=args.time_limit,
                            self_review=args.self_review,exploration_rule=args.exploration_rule,
                            feedback_reflection=args.feedback_reflection,reflection_history=args.reflection_history,
                            task_interpretation=args.task_interpretation,prediction_input=args.prediction_input,semantic_artifact_from=args.semantic_artifact_from,output=args.output)
        if code:
            return code
        source=args.prediction_input
    from superagent.format_intake import DECODE_ERRORS
    attempt = args.output.parent / 'prediction_attempt.json'
    staged = args.action == 'solve' or (args.run_dir/'first_delivery'/'prediction_manifest.json').is_file()
    try:
        if staged:
            from superagent.first_delivery import final_prediction
            path = final_prediction(source.resolve(), args.run_dir.resolve(), args.output.resolve())
        else:
            path = create_predictions(str(source), args.run_dir / 'agent', args.output)
    except DECODE_ERRORS + (RuntimeError,) as error:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        from superagent.events import RunLogger
        RunLogger(args.output.parent).write_json(attempt.name,dict(delivery_status='unavailable',
            generated_by_this_attempt=False,reason=str(error)[:500],error_type=type(error).__name__,prediction_count=None))
        print(attempt)
        return 2
    if not staged:
        from superagent.events import RunLogger
        receipt = json.loads((args.output.parent/'prediction_manifest.json').read_text(encoding='utf-8'))
        RunLogger(args.output.parent).write_json(attempt.name,dict(delivery_status=receipt.get('delivery_status','completed'),
            generated_by_this_attempt=True,prediction_count=receipt['prediction_count']))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
