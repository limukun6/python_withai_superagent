"""Train-only finalization. Freeze selection before reading validation labels."""
from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np

from .evidence_agent import route_of
from .models import SoftVoteModel, predict_with_model
from .metrics import MetricSpec,score_of

ENSEMBLE_MIN_GAIN = .002


def propose_vote(kind, history, load_model, x_search, y_search, enabled,metric=None):
    """Exactly one fixed vote; no validation input or tuned weights/thresholds."""
    record = {"enabled": enabled, "evaluations": 0, "adopted_on_inner": False,
              "minimum_gain": ENSEMBLE_MIN_GAIN, "components": [],
              "rule": "Best per visited route; max 3 routes; equal probability weights; inner gain >= .002"}
    if not enabled:
        return None, record
    completed = sorted((r for r in history if r["status"] == "completed"),
                       key=lambda r: (-score_of(r), r["candidate"]))
    winners = {}
    for row in completed:
        winners.setdefault(route_of(row["candidate"]), row)
    selected = list(winners.values())[:3]
    record["components"] = [r["candidate"] for r in selected]
    if len(selected) < 2:
        record["reason"] = "Fewer than two successfully trained routes"
        return None, record
    record["evaluations"] = 1
    try:
        vote = SoftVoteModel(kind, [load_model(r) for r in selected], np.unique(y_search))
        predicted = vote.predict(x_search)
        metric=metric or MetricSpec('accuracy',tuple(np.unique(y_search).tolist()))
        measured=metric.evaluate_model(kind,vote,x_search,y_search,predicted);score=measured['score']
        best_single = score_of(completed[0])
        record.update({'primary_metric':metric.name,'search_score':score,
                       "search_accuracy": measured['accuracy'], "best_single_search_accuracy": completed[0]['search_accuracy'],
                       'best_single_search_score':best_single,
                       "gain": score - best_single, "predictions": predicted.tolist(),
                       "adopted_on_inner": score >= best_single + ENSEMBLE_MIN_GAIN})
        return vote if record["adopted_on_inner"] else None, record
    except Exception as error:
        # Unsupported probability output is an observed failure, not a reason
        # to silently search for another better ensemble.
        record.update({"error": str(error), "reason": "Fixed vote failed; retain best single"})
        return None, record


def finalize(task, history, best_model, best_id, best_score, payload, directory,
             logger, train_candidate, start, time_limit, trial_timeout,
             refit_full_train, ensemble_selection, confirmation=None,feature_cache_dir=None):
    confirmation = confirmation or {"enabled": False}
    metric=MetricSpec.for_task(task)
    if confirmation.get("enabled") and ensemble_selection:
        raise ValueError("Confirmation and single-split ensemble selection cannot be combined")
    result = {'primary_metric':metric.name,'official_validation_score':None,'pre_refit_validation_score':None,
              'single_model_validation_score':None,"official_validation_accuracy": None, "pre_refit_validation_accuracy": None,
              "single_model_validation_accuracy": None, "refit_requested": refit_full_train,
              "refit_succeeded": False, "refit_failure": None,
              "final_training_rows": len(payload["y_train"]),
              "ensemble_selected": False, "ensemble_evaluations": 0,
              "full_refits_attempted": 0, "full_refits_completed": 0,
              'refit_classifier_fits_completed':0,
              "refit_training_seconds": 0., "ensemble_failure": None}
    if best_model is None:
        logger.write_json("selection.json", {"candidate": None, "search_accuracy": None})
        return result
    search_single = best_model
    confirmed = confirmation.get("valid_confirmation", False)
    selection_rule = confirmation["selection_rule"] if confirmed else f"max inner {metric.name}; ties by candidate id"
    selection_metric = f"confirmation_mean_{metric.name}" if confirmed else f"inner_search_{metric.name}"
    selection_score = confirmation["selection_score"] if confirmed else best_score
    def load_model(row):
        return best_model if row["candidate"] == best_id else joblib.load(directory / f"trial_{row['trial']}.joblib")["model"]
    vote, decision = propose_vote(task.kind, history, load_model, payload["X_search"],
                                  payload["y_search"], ensemble_selection,metric)
    if "predictions" in decision:
        logger.write_json("ensemble_search_predictions.json", decision.pop("predictions"))
    result["ensemble_evaluations"] = decision["evaluations"]
    best_row=next(r for r in history if r['candidate']==best_id)
    logger.write_json("selection.json", {'primary_metric':metric.name,'search_score':best_score,
        "candidate": best_id, "search_accuracy": best_row['search_accuracy'],
        "selection_rule": selection_rule, "selection_metric": selection_metric, "selection_score": selection_score,
        "refit_requested": refit_full_train, "ensemble": decision, "confirmation": confirmation})
    logger.event("selection_frozen", single=best_id, ensemble=decision, confirmation=confirmation)
    joblib.dump({"kind": task.kind, "candidate": best_id, "model": search_single},
                directory / "search_selected_model.joblib")
    singles = {best_id: search_single}
    chosen_ids = decision["components"] if vote is not None else [best_id]
    if refit_full_train:
        refit_payload = directory / "refit_data.joblib"
        # There are no provided-validation features or labels in this payload.
        first_row = task.X_train[:1]
        contents={"kind": task.kind, "mode": "refit", "X_train": task.X_train,
                     "y_train": task.y_train, "X_search": first_row,'metric_spec':metric.record()}
        from .task_semantics import declared_semantics
        contents['pair_semantics']=declared_semantics(task)
        if feature_cache_dir:contents['feature_cache_dir']=feature_cache_dir
        joblib.dump(contents,refit_payload)
        refit_success = {}
        for index, candidate in enumerate(chosen_ids):
            remaining = time_limit - (time.monotonic() - start)
            if remaining > 0:
                result["full_refits_attempted"] += 1
                fitted, metrics = train_candidate(refit_payload, candidate,
                    directory / f"refit_component_{index}.joblib", min(trial_timeout, remaining))
            else:
                fitted, metrics = None, {"status": "timeout", "error": "No time remains for final refit"}
            logger.event("final_refit", candidate=candidate, **metrics)
            result["refit_training_seconds"] += metrics.get("training_seconds", 0.)
            refit_success[candidate] = fitted is not None
            if fitted is not None:
                singles[candidate] = fitted
                result["full_refits_completed"] += 1
                result['refit_classifier_fits_completed']+=metrics.get('classifier_fits',1)
            elif candidate == best_id:
                result["refit_failure"] = metrics
        result["refit_succeeded"] = refit_success.get(best_id, False)
        if result["refit_succeeded"]:
            result["final_training_rows"] = len(task.y_train)
        if vote is not None:
            if all(refit_success.values()):
                vote = SoftVoteModel(task.kind, [singles[c] for c in chosen_ids], vote.classes_)
            else:
                vote = None
                result["ensemble_failure"] = "At least one full-train component refit failed; retain single model"
    final_model = vote if vote is not None else singles[best_id]
    result["ensemble_selected"] = vote is not None
    family = "equal_soft_vote" if vote is not None else best_id
    joblib.dump({"kind": task.kind, "candidate": family, "model": final_model}, directory / "selected_model.joblib")
    single_file = "selected_model.joblib"
    if vote is not None:
        single_file = "single_model.joblib"
        joblib.dump({"kind": task.kind, "candidate": best_id, "model": singles[best_id]}, directory / single_file)
    kinds = {"categorical_tabular": "categorical_tabular_binary", "pair_text": "pair_text_binary"}
    task_kind=kinds.get(task.kind,task.kind+'_classification')
    if task.kind=='categorical_tabular' and len(np.unique(task.y_train))>2:task_kind='categorical_tabular_multiclass'
    logger.write_json("task_spec.json", {**task.spec,'metric':metric.name,'metric_spec':metric.record(), "task_kind": task_kind,"model_kind":task.kind,
                                       "class_count":int(len(np.unique(task.y_train)))})
    logger.write_json("final_selection.json", {
        "selected_experiment": "ensemble" if vote is not None else f"trial_{next(r['trial'] for r in history if r['candidate'] == best_id)}",
        "family": family, "single_candidate": best_id, "single_model_file": single_file,
        'primary_metric':metric.name,"selection_metric": selection_metric,
        "selection_score": decision["search_score"] if vote is not None else selection_score,
        "model_file": "selected_model.joblib", "model_format": "candidate_bundle_v1",
        "ensemble_components": chosen_ids if vote is not None else [],
        "test_accessed": False, "refit": result["refit_succeeded"], "refit_requested": refit_full_train,
        "training_rows": result["final_training_rows"],
        "model_sha256": hashlib.sha256((directory / "selected_model.joblib").read_bytes()).hexdigest(),
        "task_spec_sha256":hashlib.sha256((directory/'task_spec.json').read_bytes()).hexdigest(),
    })
    logger.event("final_model_frozen", family=family, ensemble_selected=vote is not None)
    if task.X_val is None or task.y_val is None:
        if not(task.X_val is None and task.y_val is None):raise ValueError('External validation features and labels must both be present or absent')
        logger.event('validation_unavailable',reason='No designated validation split supplied; internal selection only')
        result['provided_validation_available']=False
        return result
    # All model choices and execution-failure fallbacks above precede scoring.
    predicted = predict_with_model(task.kind, final_model, task.X_val)
    single_prediction = predicted if vote is None else predict_with_model(task.kind, singles[best_id], task.X_val)
    try:
        measured=metric.evaluate_model(task.kind,final_model,task.X_val,task.y_val,predicted)
        single=metric.evaluate_model(task.kind,singles[best_id],task.X_val,task.y_val,single_prediction)
    except ValueError as error:
        result['provided_validation_available']=True
        result['validation_failure']={'type':type(error).__name__,'message':str(error)}
        logger.event('validation_failed',**result['validation_failure'])
        return result
    result['official_validation_metrics']=measured
    result['official_validation_score']=measured['score'];result['single_model_validation_score']=single['score']
    result["official_validation_accuracy"] = measured['accuracy']
    result["single_model_validation_accuracy"] = single['accuracy']
    logger.write_json("validation_predictions.json", predicted.tolist())
    logger.write_json("single_validation_predictions.json", single_prediction.tolist())
    if refit_full_train:
        before = predict_with_model(task.kind, search_single, task.X_val)
        before_metrics=metric.evaluate_model(task.kind,search_single,task.X_val,task.y_val,before)
        result['pre_refit_validation_score']=before_metrics['score']
        result["pre_refit_validation_accuracy"] = before_metrics['accuracy']
        logger.write_json("pre_refit_validation_predictions.json", before.tolist())
    logger.event("validation_scored",primary_metric=metric.name,score=result['official_validation_score'],accuracy=result["official_validation_accuracy"],
                 single_accuracy=result["single_model_validation_accuracy"])
    return result
