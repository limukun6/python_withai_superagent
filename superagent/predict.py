"""Frozen-model prediction stage.  It intentionally has no test evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .data import load_prediction_features
from .events import RunLogger, atomic_text
from .models import predict_with_model,probability_with_model
from .metrics import MetricSpec


class PredictionAdaptationError(ValueError):
    """Only input or output-source preflight failed; the frozen model remains valid."""
    pass


def validate_prediction_outputs(predictions, rows, specification, classes=None, scores=None):
    """Check unlabeled delivery against the frozen training contract before writing."""
    predictions=np.asarray(predictions)
    if predictions.ndim!=1 or len(predictions)!=rows:
        raise ValueError('Prediction output must contain one scalar per input row')
    values=predictions.tolist()
    if any(type(v) not in (str,int,float,bool) or isinstance(v,float) and not np.isfinite(v) for v in values):
        raise ValueError('Predicted labels must be finite scalar values')
    declared=specification.get('metric_spec')
    metric=MetricSpec.from_payload({'metric_spec':declared}) if declared is not None else None
    if metric is not None:
        if metric.name!=specification.get('metric','accuracy'):raise ValueError('Frozen metric declarations disagree')
        if not set(values)<=set(metric.labels):raise ValueError('Prediction contains a class outside the frozen training domain')
    from .output_recipe import needs_scores
    require_scores = specification.get('metric','accuracy')=='roc_auc' or needs_scores(specification.get('output_recipe'))
    if require_scores:
        if metric is None:raise ValueError('Probability export requires a frozen metric class domain')
        order=np.asarray(classes)
        matrix=np.asarray(scores,dtype=float)
        if (order.ndim!=1 or matrix.shape!=(rows,len(metric.labels)) or len(order)!=len(metric.labels)
                or len(set(order.tolist()))!=len(order) or set(order.tolist())!=set(metric.labels)
                or not np.isfinite(matrix).all()):raise ValueError('Prediction probability rows or class columns disagree with frozen contract')
        if np.any(matrix<0) or np.any(matrix>1) or not np.allclose(matrix.sum(axis=1),1.,atol=1e-6):
            raise ValueError('Prediction probabilities must be normalized')
        classes,scores=order,matrix
    return predictions,classes,scores,{'row_count_checked':True,'label_domain_checked':metric is not None,
                                      'probability_contract_checked':require_scores}


def create_predictions(task_package: str, run_dir: str | Path, output_path: str | Path, *, prediction_adapter=None) -> Path:
    run_path = Path(run_dir).resolve()
    selection = json.loads((run_path / "final_selection.json").read_text(encoding="utf-8"))
    specification = json.loads((run_path / "task_spec.json").read_text(encoding="utf-8"))
    if selection.get('task_spec_sha256') and hashlib.sha256((run_path/'task_spec.json').read_bytes()).hexdigest()!=selection['task_spec_sha256']:
        raise ValueError('Task schema no longer matches the frozen selection hash')
    if selection.get("test_accessed"):
        raise RuntimeError("This run directory is already marked as having accessed test data.")
    if selection.get('model_format')=='class_prior_v1':
        from .partial_classification import predict
        return predict(task_package,run_path,output_path,selection)
    kind = specification["task_kind"]
    kind_to_loader = {
        "categorical_tabular_binary": "categorical_tabular",
        "pair_text_binary": "pair_text",
        "categorical_tabular_multiclass":"categorical_tabular",
        **{kind+'_classification':kind for kind in ('categorical_tabular','pair_text','numeric_tabular','mixed_tabular','text_tabular','single_text','image')},
    }
    if kind not in kind_to_loader:
        raise ValueError(f"Unsupported frozen task kind: {kind}")
    model_path=run_path/selection['model_file']
    if selection.get('model_sha256') and hashlib.sha256(model_path.read_bytes()).hexdigest()!=selection['model_sha256']:
        raise ValueError('Model file no longer matches the frozen selection hash')
    model = joblib.load(model_path)
    model_format = selection.get("model_format", "bare_model")
    if model_format == "candidate_bundle_v1":
        family = selection["family"]
        if family == 'class_pair_refinement':
            from .confusion_specialist import ClassPairModel
            fitted = model.get('model') if isinstance(model,dict) else None
            if isinstance(fitted,ClassPairModel):
                family = fitted.verified_legacy_id(selection)
        if (not isinstance(model, dict) or "model" not in model or not family
            or model.get("kind") != kind_to_loader[kind] or model.get("candidate") != family):
            raise ValueError("Model bundle does not match its frozen task and selection.")
        model = model["model"]
    elif model_format == 'verified_trial_v1':
        metric = MetricSpec.from_payload({'metric_spec':specification['metric_spec']})
        if (not isinstance(model,dict) or model.get('metrics',{}).get('resolved_configuration',{}).get('id') != selection['family']
                or metric.check_identity(model['metrics'].get('metric_spec'))['status'] != 'confirmed'
                or model.get('prediction_record',{}).get('evaluation_payload_sha256') != selection['evaluation_payload_sha256']):
            raise ValueError('Verified partial trial differs from the frozen selection')
        model = model['model']
    elif model_format != "bare_model":
        raise ValueError(f"Unsupported model format: {model_format}")
    if prediction_adapter is not None:
        from .prediction_adapter import load_adapter
        specification=load_adapter(prediction_adapter,task_package,run_path,specification,selection)
    from .format_intake import DECODE_ERRORS
    try:
        features = load_prediction_features(task_package, kind_to_loader[kind], specification)
        if specification.get('output_recipe') is not None:
            from .output_recipe import probe_output_context
            probe_output_context(task_package,specification,len(features))
    except DECODE_ERRORS as error:
        from .semantic_contract import _diagnostic
        diagnostic=_diagnostic(error,'prediction_delivery')
        raise PredictionAdaptationError('Prediction preflight: '+diagnostic['message']) from error
    predictions = predict_with_model(kind_to_loader[kind], model, features)
    scores=classes=None
    from .output_recipe import needs_scores, write_export
    if specification.get('metric','accuracy')=='roc_auc' or needs_scores(specification.get('output_recipe')):
        classes,scores=probability_with_model(kind_to_loader[kind],model,features)
    predictions,classes,scores,contract_check=validate_prediction_outputs(predictions,len(features),specification,classes,scores)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_contract = None
    if specification.get('output_recipe') is not None:
        output_contract = write_export(task_package, specification, destination, predictions, classes, scores)
    else:
        with atomic_text(destination, newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["row_id", "prediction"]+([f'score_{i}' for i in range(len(classes))] if scores is not None else []))
            writer.writerows([i,value,*scores[i].tolist()] if scores is not None else [i,value] for i,value in enumerate(predictions.tolist()))
    manifest = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "selected_experiment": selection["selected_experiment"],
        "prediction_count": int(len(predictions)),
        "delivery_status": selection.get('delivery_status','completed'),
        "source_trial": selection.get('source_trial'), "refit": selection.get('refit'),
        "partial_reason": selection.get('partial_reason'),
        "input_source": str(Path(task_package).resolve()),
        "test_labels_read": False,
        "prediction_file": str(destination),
        "prediction_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        'score_class_order':classes.tolist() if classes is not None else None,
        'prediction_contract':contract_check,
        'output_contract':output_contract,
        'prediction_adapter_sha256':hashlib.sha256(Path(prediction_adapter).read_bytes()).hexdigest() if prediction_adapter is not None else None,
    }
    RunLogger(destination.parent).write_json("prediction_manifest.json",manifest)
    return destination
