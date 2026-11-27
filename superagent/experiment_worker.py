"""Isolated training worker; receives only inner training/search data."""
from __future__ import annotations

import sys
import hashlib
import io
import json
from pathlib import Path
import time
import joblib
from sklearn.metrics import accuracy_score, recall_score
from .models import search_candidates, predict_with_model, probability_with_model
from .feature_cache import using_feature_cache
from .prediction_evidence import make_prediction_record
from .metrics import MetricSpec
from .error_slices import ErrorSlices


def main() -> None:
    arguments = sys.argv[1:]
    fit_admission = None
    if len(arguments) >= 2 and arguments[-2] == '--fit-admission':
        fit_admission = json.loads(Path(arguments[-1]).read_text(encoding='utf-8'))
        arguments = arguments[:-2]
    if len(arguments) not in (3,4):raise ValueError('Expected payload, candidate, output, and optional owned registry')
    payload_path, candidate_id, output_path = arguments[:3]
    payload_bytes=Path(payload_path).read_bytes()
    payload = joblib.load(io.BytesIO(payload_bytes))
    payload_sha256=hashlib.sha256(payload_bytes).hexdigest()
    metric=MetricSpec.from_payload(payload)
    registry_sha256=None;origin='static'
    if len(arguments)==4:
        from .candidate_refinement import resolve_registered
        registry_bytes=Path(arguments[3]).read_bytes();document=json.loads(registry_bytes)
        candidate=resolve_registered(payload['kind'],candidate_id,document)
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest()
        if candidate_id in document['entries']:origin='registry'
    else:candidate = next(c for c in search_candidates(payload["kind"], "all") if c.family == candidate_id)
    from .task_semantics import validate_candidate_semantics
    semantic_check=validate_candidate_semantics(payload['kind'],payload.get('pair_semantics'),candidate)
    configuration={'id':candidate.family,'params':candidate.params,'origin':origin,'registry_sha256':registry_sha256,
                   'semantic_check':semantic_check}
    start = time.perf_counter()
    from .fit_guard import admitted_fits
    from .training_randomness import fitted_seed, training_seed
    repeat_seed = payload.get("supervised_random_seed")
    runtime_estimator=None
    if candidate.params.get('runtime_composed') and origin=='registry':
        from .runtime_inventory import runtime_classifiers
        runtime_estimator=runtime_classifiers()[candidate.params['class']]
    with admitted_fits(fit_admission, candidate_id, runtime_estimator=runtime_estimator) as fit_counts, using_feature_cache(payload.get('feature_cache_dir')) as cache, training_seed(repeat_seed):
        model, predictions = candidate.fit_predict(payload["X_train"], payload["y_train"], payload["X_search"])
        if repeat_seed is not None:
            if fitted_seed(model) != repeat_seed:
                raise ValueError("Fitted classifier did not use the requested replicate seed")
            configuration["training_random_seed"] = repeat_seed
        if payload.get("mode") != "refit":
            training_predictions = predict_with_model(payload["kind"], model, payload["X_train"])
        cache_report=cache.report() if cache else {'enabled':False}
    classifier_fits = fit_counts['completed_classifier_fits'] if fit_counts is not None else int(getattr(model,'classifier_fits',1))
    if fit_counts is not None and int(getattr(model,'classifier_fits',classifier_fits)) != classifier_fits:
        raise ValueError('Model fit declaration differs from actual supervised calls')
    training_metadata=getattr(model,'training_metadata',None)
    selected_iterations=training_metadata.get('selected_iterations') if training_metadata else None
    from .estimator_facts import observed_classifier
    observed=observed_classifier(model)
    if runtime_estimator is not None:
        if (observed['class'],observed['module'])!=(runtime_estimator.__name__,runtime_estimator.__module__):
            raise ValueError('The fitted estimator differs from the generated pipeline')
        observed['family']=observed['family'] or observed['class']
    if payload.get("mode") == "refit":
        joblib.dump({"model": model, "metrics": {"training_rows": len(payload["y_train"]),
                     "training_seconds": time.perf_counter() - start,'feature_cache':cache_report,
                     'classifier_fits':classifier_fits,
                     'training_metadata':training_metadata,'selected_iterations':selected_iterations,
                     'estimator':observed,'resolved_configuration':configuration,'metric_spec':metric.record()}}, output_path)
        return
    train_metrics=metric.evaluate_model(payload['kind'],model,payload['X_train'],payload['y_train'],training_predictions)
    search_metrics=metric.evaluate_model(payload['kind'],model,payload['X_search'],payload['y_search'],predictions)
    score_evidence={}
    if payload.get('preference_vote_probabilities'):
        from .ab_complement import canonical_probabilities
        classes,scores=canonical_probabilities(model,payload['X_search'])
        score_evidence={'search_scores':scores,'score_classes':classes}
    elif metric.requires_scores or payload.get('capture_probabilities'):
        try:
            classes,scores=probability_with_model(payload['kind'],model,payload['X_search'])
            score_evidence={'search_scores':scores,'score_classes':classes}
        except (AttributeError, NotImplementedError, ValueError) as error:
            if metric.requires_scores:
                raise
            configuration['probability_unavailable'] = type(error).__name__
    prediction_record=make_prediction_record(predictions,payload['y_search'],payload_sha256)
    if score_evidence:
        from .score_evidence import make_score_record
        prediction_record['score_record']=make_score_record(prediction_record,scores,classes,metric,candidate_id)
    train_accuracy=train_metrics["accuracy"];search_accuracy=search_metrics["accuracy"]
    slices = ErrorSlices(payload["kind"], payload["X_train"]).describe(payload["X_search"], payload["y_search"], predictions) if payload.get("error_feedback") else None
    joblib.dump({"model": model if payload.get("retain_model", True) else None,
                **score_evidence,
                "search_predictions": predictions,
                'prediction_record':prediction_record, "metrics": {
        "error_slices":slices, "primary_metric":metric.name,'metric_spec':metric.record(),"train_score":train_metrics["score"],"search_score":search_metrics["score"],
        "train_metrics":train_metrics,"search_metrics":search_metrics,"score_gap":train_metrics["score"]-search_metrics["score"],
        "train_accuracy": train_accuracy,
        "search_accuracy": search_accuracy,
        "generalization_gap": train_accuracy - search_accuracy,
        "class_recall": recall_score(payload["y_search"], predictions, average=None, zero_division=0).tolist(),
        "training_seconds": time.perf_counter() - start,
        'feature_cache':cache_report,
        'classifier_fits':classifier_fits,
        'training_metadata':training_metadata,'selected_iterations':selected_iterations,
        'estimator':observed,'resolved_configuration':configuration,
    }}, output_path)


if __name__ == "__main__":
    main()
