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
    if len(sys.argv) not in (4,5):raise ValueError('Expected payload, candidate, output, and optional owned registry')
    payload_path, candidate_id, output_path = sys.argv[1:4]
    payload_bytes=Path(payload_path).read_bytes()
    payload = joblib.load(io.BytesIO(payload_bytes))
    payload_sha256=hashlib.sha256(payload_bytes).hexdigest()
    metric=MetricSpec.from_payload(payload)
    registry_sha256=None;origin='static'
    if len(sys.argv)==5:
        from .candidate_refinement import resolve_registered
        registry_bytes=Path(sys.argv[4]).read_bytes();document=json.loads(registry_bytes)
        candidate=resolve_registered(payload['kind'],candidate_id,document)
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest()
        if candidate_id in document['entries']:origin='registry'
    else:candidate = next(c for c in search_candidates(payload["kind"], "all") if c.family == candidate_id)
    from .task_semantics import validate_candidate_semantics
    semantic_check=validate_candidate_semantics(payload['kind'],payload.get('pair_semantics'),candidate)
    configuration={'id':candidate.family,'params':candidate.params,'origin':origin,'registry_sha256':registry_sha256,
                   'semantic_check':semantic_check}
    start = time.perf_counter()
    with using_feature_cache(payload.get('feature_cache_dir')) as cache:
        model, predictions = candidate.fit_predict(payload["X_train"], payload["y_train"], payload["X_search"])
        if payload.get("mode") != "refit":
            training_predictions = predict_with_model(payload["kind"], model, payload["X_train"])
        cache_report=cache.report() if cache else {'enabled':False}
    training_metadata=getattr(model,'training_metadata',None)
    selected_iterations=training_metadata.get('selected_iterations') if training_metadata else None
    from .estimator_facts import observed_classifier
    observed=observed_classifier(model)
    if payload.get("mode") == "refit":
        joblib.dump({"model": model, "metrics": {"training_rows": len(payload["y_train"]),
                     "training_seconds": time.perf_counter() - start,'feature_cache':cache_report,
                     'classifier_fits':int(getattr(model,'classifier_fits',1)),
                     'training_metadata':training_metadata,'selected_iterations':selected_iterations,
                     'estimator':observed,'resolved_configuration':configuration}}, output_path)
        return
    train_metrics=metric.evaluate_model(payload['kind'],model,payload['X_train'],payload['y_train'],training_predictions)
    search_metrics=metric.evaluate_model(payload['kind'],model,payload['X_search'],payload['y_search'],predictions)
    score_evidence={}
    if metric.requires_scores:
        classes,scores=probability_with_model(payload['kind'],model,payload['X_search'])
        score_evidence={'search_scores':scores,'score_classes':classes}
    train_accuracy=train_metrics["accuracy"];search_accuracy=search_metrics["accuracy"]
    slices = ErrorSlices(payload["kind"], payload["X_train"]).describe(payload["X_search"], payload["y_search"], predictions) if payload.get("error_feedback") else None
    joblib.dump({"model": model if payload.get("retain_model", True) else None,
                **score_evidence,
                "search_predictions": predictions,
                'prediction_record':make_prediction_record(predictions,payload['y_search'],payload_sha256), "metrics": {
        "error_slices":slices, "primary_metric":metric.name,"train_score":train_metrics["score"],"search_score":search_metrics["score"],
        "train_metrics":train_metrics,"search_metrics":search_metrics,"score_gap":train_metrics["score"]-search_metrics["score"],
        "train_accuracy": train_accuracy,
        "search_accuracy": search_accuracy,
        "generalization_gap": train_accuracy - search_accuracy,
        "class_recall": recall_score(payload["y_search"], predictions, average=None, zero_division=0).tolist(),
        "training_seconds": time.perf_counter() - start,
        'feature_cache':cache_report,
        'classifier_fits':int(getattr(model,'classifier_fits',1)),
        'training_metadata':training_metadata,'selected_iterations':selected_iterations,
        'estimator':observed,'resolved_configuration':configuration,
    }}, output_path)


if __name__ == "__main__":
    main()
