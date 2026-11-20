"""Frozen-model prediction stage.  It intentionally has no test evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import load_prediction_features
from .models import predict_with_model,probability_with_model
from .metrics import MetricSpec


def create_predictions(task_package: str, run_dir: str | Path, output_path: str | Path, minilm_weights=None, deep_minilm_weights=None) -> Path:
    run_path = Path(run_dir).resolve()
    selection = json.loads((run_path / "final_selection.json").read_text(encoding="utf-8"))
    specification = json.loads((run_path / "task_spec.json").read_text(encoding="utf-8"))
    if selection.get('task_spec_sha256') and hashlib.sha256((run_path/'task_spec.json').read_bytes()).hexdigest()!=selection['task_spec_sha256']:
        raise ValueError('Task schema no longer matches the frozen selection hash')
    if selection.get("test_accessed"):
        raise RuntimeError("This run directory is already marked as having accessed test data.")
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
        if (not isinstance(model, dict) or "model" not in model
            or model.get("kind") != kind_to_loader[kind] or model.get("candidate") != selection["family"]):
            raise ValueError("Model bundle does not match its frozen task and selection.")
        model = model["model"]
    elif model_format != "bare_model":
        raise ValueError(f"Unsupported model format: {model_format}")
    destination = Path(output_path).resolve()
    feature_dir = None
    from .minilm_features import MiniLMTextClassifier, feature_usage
    from .encoder_mixture import EncoderMixture, bind_text_prediction
    if isinstance(model, (MiniLMTextClassifier,EncoderMixture)):
        config = run_path / 'pretrained_runtime.json'
        settings = json.loads(config.read_text()) if config.exists() else {}
        if minilm_weights is not None:
            setting = model.weight_setting if isinstance(model,MiniLMTextClassifier) else 'minilm_weights'
            settings[setting] = minilm_weights
        if deep_minilm_weights is not None:settings['deep_minilm_weights'] = deep_minilm_weights
        feature_dir = destination.parent / (destination.name + '.features')
        bind_text_prediction(model,settings,feature_dir)
        before = feature_usage(feature_dir)
    features = load_prediction_features(task_package, kind_to_loader[kind], specification)
    predictions = predict_with_model(kind_to_loader[kind], model, features)
    scores=classes=None
    if specification.get('metric','accuracy')=='roc_auc':classes,scores=probability_with_model(kind_to_loader[kind],model,features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "prediction"]+([f'score_{i}' for i in range(len(classes))] if scores is not None else []))
        writer.writerows([i,value,*scores[i].tolist()] if scores is not None else [i,value] for i,value in enumerate(predictions.tolist()))
    manifest = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "selected_experiment": selection["selected_experiment"],
        "prediction_count": int(len(predictions)),
        "input_source": str(Path(task_package).resolve()),
        "test_labels_read": False,
        "prediction_file": str(destination),
        'score_class_order':classes.tolist() if classes is not None else None,
    }
    if feature_dir is not None:
        after = feature_usage(feature_dir)
        manifest['prediction_feature_usage'] = {k:after[k]-before[k] for k in after}
    (destination.parent / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
