"""Post-search confirmation using training data alone; no planner or API calls."""
import math
import statistics
import time

import joblib

from .splits import effective_grouping, inner_split, grouped_partition, overlap_audit, take
from .metrics import score_of,completed,metric_of,MetricSpec


def shortlist(history, top_k=3):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    metric_of(history)
    eligible = [r for r in history if completed(r)]
    if len({r["candidate"] for r in eligible}) != len(eligible):
        raise ValueError("Repeated candidate IDs are ambiguous")
    return sorted(eligible, key=lambda r: (-score_of(r), r["candidate"]))[:top_k]


def decide(primary, measurements, seeds, grouping):
    """Any missing result invalidates confirmation, preserving the old winner."""
    name=metric_of(primary)
    record = {'primary_metric':name,"enabled": True, "grouping": grouping, "seeds": list(seeds),
              "shortlist": [r["candidate"] for r in primary], "primary_candidate": primary[0]["candidate"] if primary else None,
              "selection_rule": f"mean {name} across all confirmation splits; ties by candidate id",
              "candidates": [], "valid_confirmation": False, "selected_candidate": primary[0]["candidate"] if primary else None}
    if not primary:
        record["failure"] = "No completed search candidates"
        return record
    for row in primary:
        values = measurements.get(row["candidate"], [])
        valid = (len(values) == len(seeds) and [v["seed"] for v in values] == list(seeds)
                 and all(completed(v) and v.get('primary_metric','accuracy')==name for v in values))
        scores = [score_of(v) for v in values] if valid else []
        accuracies=[v['search_accuracy'] for v in values] if valid else []
        record["candidates"].append({"candidate": row["candidate"], "primary_accuracy": row["search_accuracy"],
            "measurements": values, "complete": valid,
            'primary_score':score_of(row),'mean_score':statistics.mean(scores) if valid else None,
            "mean_accuracy": statistics.mean(accuracies) if valid else None,
            "sample_sd": statistics.stdev(scores) if len(scores) > 1 else None})
    if not all(r["complete"] for r in record["candidates"]):
        record["failure"] = "Incomplete confirmation; keep the primary winner, never treat missing scores as zero"
        return record
    best = min(record["candidates"], key=lambda r: (-r["mean_score"], r["candidate"]))
    record.update({"valid_confirmation": True, "selected_candidate": best["candidate"],
                   "selection_score": best["mean_score"], "changed": best["candidate"] != primary[0]["candidate"]})
    return record


def confirm_candidates(task, history, seed, directory, logger, train_candidate, start,
                       time_limit, trial_timeout, repeats=3, top_k=3, grouping="exact_rows", primary_grouping="exact_rows",feature_cache_dir=None):
    if repeats not in (1,3):
        raise ValueError("Confirmation supports either disabled (1) or three prespecified splits")
    if repeats == 1:
        return {"enabled": False}
    selected = shortlist(history, top_k)
    metric=MetricSpec.for_task(task)
    for row in selected:
        if row.get('primary_metric','accuracy')!=metric.name:raise ValueError('Confirmation history uses a different task metric')
        metric.check_identity(row.get('metric_spec'))
    seeds = [seed + 10000*i for i in range(repeats)]
    mode = effective_grouping(task.kind, grouping)
    primary_mode = effective_grouping(task.kind, primary_grouping)
    records = {r["candidate"]: [] for r in selected}
    fits = completed = 0
    classifier_fits_completed=0
    training_seconds = 0.
    for index, split_seed in enumerate(seeds):
        if index == 0 and mode == primary_mode:
            for row in selected:
                records[row["candidate"]].append({"seed": split_seed, "status": "completed",
                    'primary_metric':metric.name,'search_score':score_of(row),
                    'metric_spec':row.get('metric_spec'),'metric_identity':metric.check_identity(row.get('metric_spec')),
                    "search_accuracy": row["search_accuracy"], "origin": "primary_search", "new_fit": False})
            continue
        try:
            train, search, split_resource = grouped_partition(task, split_seed, mode)
        except Exception as error:
            for row in selected:
                records[row["candidate"]].append({"seed": split_seed, "status": "failed", "error": str(error), "new_fit": False})
            continue
        logger.write_json(f"confirmation_split_{index}.json", {"seed": split_seed, "grouping": mode,
            'split_resource':split_resource,
            "train_indices": train.tolist(), "search_indices": search.tolist(), **overlap_audit(task, train, search)})
        payload = directory / f"confirmation_data_{index}.joblib"
        contents={"kind": task.kind, "X_train": take(task.X_train, train), "y_train": task.y_train[train],
                     "X_search": take(task.X_train, search), "y_search": task.y_train[search], "retain_model": False,
                     'metric_spec':metric.record()}
        from .task_semantics import declared_semantics
        contents['pair_semantics']=declared_semantics(task)
        if feature_cache_dir:contents['feature_cache_dir']=feature_cache_dir
        joblib.dump(contents,payload)
        for row in selected:
            remaining = time_limit - (time.monotonic() - start)
            metrics = {"status": "timeout", "error": "No time remains for confirmation", "new_fit": False}
            if remaining > 0:
                fits += 1
                _, metrics = train_candidate(payload, row["candidate"],
                    directory / f"confirmation_{index}_{row['trial']}.joblib", min(trial_timeout, remaining))
                metrics["new_fit"] = True
                metrics['metric_identity']=metric.check_identity(metrics.get('metric_spec'))
                completed += metrics["status"] == "completed"
                if metrics['status']=='completed':classifier_fits_completed+=metrics.get('classifier_fits',1)
                training_seconds += metrics.get("training_seconds", 0.)
            records[row["candidate"]].append({"seed": split_seed, "origin": "confirmation", **metrics})
            logger.event("confirmation_result", candidate=row["candidate"], seed=split_seed, **metrics)
    decision = decide(selected, records, seeds, mode)
    decision.update({"new_fits_attempted": fits, "new_fits_completed": completed, "training_seconds": training_seconds,
        'classifier_fits_completed':classifier_fits_completed,
        'failed_attempts_with_unknown_component_count':fits-completed})
    logger.write_json("confirmation.json", decision)
    logger.event("confirmation_selected", **decision)
    return decision
