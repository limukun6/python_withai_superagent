"""Bind historical executed/used split IDs, then recover evaluation overlap from training rows."""
import argparse
from pathlib import Path

from scripts.unified_released_study import ROOT, archive, read, require, save, sha
from superagent.ab_run import confirmation_coverage
from superagent.ab_scheduler import PRIMARY, CONFIRMATIONS
from superagent.data import TrainingTask, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups, take
from unified_agent import audit_partition


def capture(runs):
    result = {}
    for tag in ("table","titles"):
        path = runs/tag/"run"/"agent"
        history, decision = read(path/"history.json"), read(path/"confirmation.json")
        receipt = read(ROOT/"research/unified_raw_1124"/tag/"completion.json")
        require(sha(path/"selected_model.joblib")==receipt["model"],"Unexpected historical model")
        used = decision["b_scheduler_review"].get("basis")
        if used is None:
            used = [PRIMARY,CONFIRMATIONS[0]] if decision["b_scheduler_review"]["confirmation_rows"] else [PRIMARY]
        executed = {r["split_id"] for r in history if r["status"]=="completed"}
        splits = {}
        for name in (PRIMARY,*CONFIRMATIONS):
            if name in executed:
                packet = read(path/(name+"_split.json"))
                require(fingerprint(packet["search_indices"])==packet["row_ids_sha256"],"Historical rows changed")
                splits[name] = {"seed":packet["seed"],"evaluation_rows_sha256":packet["row_ids_sha256"],
                                "training_rows_sha256":fingerprint(packet["train_indices"])}
        result[tag] = dict(model=receipt["model"],history_sha256=sha(path/"history.json"),
                           confirmation_sha256=sha(path/"confirmation.json"), used_for_selection=used,
                           as_recorded_independent=decision["independent_confirmation_observed"],completed=splits)
    return result


def reconstruct(packages, trace):
    results = {}
    for tag,record in trace.items():
        task = load_training_task(archive(packages,tag))
        audit,_ = audit_partition(task)
        if audit is not None:
            fit = audit[0]
            task = TrainingTask(task.kind, take(task.X_train,fit), task.y_train[fit], task.spec)
        groups = sample_groups(task,"shared_text")
        trusted = {}
        for name,packet in record["completed"].items():
            train,held,_ = grouped_partition(task,packet["seed"],"shared_text")
            require(fingerprint(train.tolist())==packet["training_rows_sha256"] and
                    fingerprint(held.tolist())==packet["evaluation_rows_sha256"],"Historical split could not be reconstructed")
            require(not set(groups[train]).intersection(groups[held]),"Training group found in the same trial's evaluation")
            trusted[name] = dict(row_ids=held.tolist(),groups=groups[held].tolist())
        coverage = confirmation_coverage(trusted,trusted,record["used_for_selection"])
        results[tag] = dict(model_training_rows=len(task.y_train),within_split_shared_groups=0,
                            historical_flag=record["as_recorded_independent"],**coverage)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage",required=True)
    c = sub.add_parser("capture")
    c.add_argument("runs",type=Path)
    v = sub.add_parser("verify")
    v.add_argument("packages",type=Path)
    v.add_argument("trace",type=Path)
    for cmd in (c,v):
        cmd.add_argument("output",type=Path)
    a = p.parse_args()
    save(a.output,capture(a.runs) if a.stage=="capture" else reconstruct(a.packages,read(a.trace)))
