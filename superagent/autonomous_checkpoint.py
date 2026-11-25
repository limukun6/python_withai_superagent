"""Keep an independently usable, explicitly partial delivery while the autonomous search runs."""
import json
from pathlib import Path

import joblib
import numpy as np

from .ab_run import sha
from .models import predict_with_model, probability_with_model
from .partial_classification import class_prior
from .splits import take


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_spec(task, spec, metric):
    return dict(task.spec,task_kind=task.kind+'_classification',model_kind=task.kind,class_count=len(metric.labels),
                metric=metric.name,metric_spec=metric.record(),goal_resolution=spec['goal_resolution'])


def standby(task, spec, metric, directory, log):
    schema = frozen_spec(task,spec,metric)
    log.write_json('task_spec.json',schema)
    stats = class_prior(task.y_train,metric.name,task.spec.get('positive_label'))
    fmt = 'adaptive_records' if schema.get('input_recipe') else 'frozen_features'
    prior = dict(stats, format=fmt, target=schema.get('input_schema',{}).get('target'), frozen_spec=schema,
                 model_kind=task.kind)
    log.write_json('partial_model.json',prior)
    selection = dict(selected_experiment='training_class_prior',family='training_class_prior',
        delivery_status='partial',requested_planner_completed=False,contract_unresolved=False,
        partial_reason='Search or final refit has not completed',model_format='class_prior_v1',
        model_file='partial_model.json',model_sha256=sha(directory/'partial_model.json'),
        task_spec_sha256=sha(directory/'task_spec.json'),primary_metric=metric.name,
        refit=False,test_accessed=False,training_rows=len(task.y_train),full_training_rows=len(task.y_train))
    log.write_json('final_selection.json',selection)
    log.write_json('prior_selection.json',selection)
    return selection


def promote(task, metric, prior, trial, candidate, model_path, payload_record, observation, log, *, review_evidence=None):
    if review_evidence is None and prior.get('selection_score',float('-inf')) >= observation['score']:
        return prior
    bundle = joblib.load(model_path)
    payload = joblib.load(payload_record['path'])
    if sha(payload_record['path']) != payload_record['sha256']:
        raise ValueError('Primary partition changed before checkpoint publication')
    indexes = np.linspace(0,len(observation['predictions'])-1,min(8,len(observation['predictions'])),dtype=int)
    probe = take(payload['X_search'],indexes)
    predicted = predict_with_model(task.kind,bundle['model'],probe)
    np.testing.assert_array_equal(predicted,observation['predictions'][indexes])
    if metric.requires_scores:
        classes, scores = probability_with_model(task.kind,bundle['model'],probe)
        order = list(classes)
        np.testing.assert_allclose(scores[:,[order.index(c) for c in metric.labels]],
                                   observation['probabilities'][indexes],rtol=1e-6,atol=1e-8)
    selection = dict(selected_experiment='verified_primary_'+str(trial),family=candidate,source_trial=trial,
        single_candidate=candidate,ensemble_components=[],model_format='verified_trial_v1',model_file=model_path.name,
        model_sha256=sha(model_path),task_spec_sha256=prior['task_spec_sha256'],
        primary_metric=metric.name,selection_metric='verified primary training holdout',selection_score=observation['score'],
        evaluation_payload_sha256=payload_record['sha256'],model_replay_rows=len(indexes),
        training_rows=len(payload['y_train']),full_training_rows=len(task.y_train),refit=False,test_accessed=False,
        delivery_status='partial',requested_planner_completed=False,contract_unresolved=False,
        partial_reason='Search or final refit has not completed')
    if review_evidence is not None:
        selection.update(selection_metric='row-weighted gain across three training folds',
                         selection_score=review_evidence['weighted_delta'],selection_evidence='scheduled_review.json',
                         primary_score=observation['score'],previous_model_sha256=prior['model_sha256'])
    log.write_json('verified_selection.json',selection)
    log.write_json('final_selection.json',selection)
    return selection


def interrupted(directory, reason):
    """Validate the published pointer and retain known/unknown worker usage after a child exits."""
    from .events import RunLogger
    log = RunLogger(directory)
    pointer = directory/'final_selection.json'
    def intact(path):
        try:
            selected = read(path)
            name = selected.get('model_file','')
            if (name and Path(name).name == name and selected.get('test_accessed') is False
                    and sha(directory/name) == selected.get('model_sha256')
                    and sha(directory/'task_spec.json') == selected.get('task_spec_sha256')):
                return selected
        except (ValueError, OSError, TypeError, AttributeError):
            return None
    selected = intact(pointer)
    if selected and selected.get('model_format')=='class_prior_v1':
        selected = intact(directory/'verified_selection.json') or selected
    if not selected:
        selected = intact(directory/'verified_selection.json') or intact(directory/'prior_selection.json')
    if not selected:
        return False
    if selected.get('delivery_status') != 'partial':
        try:
            completed_run = read(directory/'run_outcome.json').get('status') == 'completed'
        except (OSError, ValueError, AttributeError):
            completed_run = False
        if completed_run:
            return True
        selected=dict(selected,delivery_status='partial',requested_planner_completed=False,
                      full_training_rows=selected['training_rows'])
    admissions = sorted(directory.glob('*.admission.json'))
    attempted = completed = unknown = 0
    for admission in admissions:
        ledger = admission.with_name(admission.name.removesuffix('.admission.json')+'.fits.json')
        try:
            record = read(ledger)
        except (OSError, ValueError):
            unknown += 1
            continue
        attempted += record.get('attempted_classifier_fits',0)
        completed += record.get('completed_classifier_fits',0)
        unknown += int(record.get('status') not in ('completed','failed') or bool(record.get('usage_unknown')))
    def journal(path):
        rows=[]
        if path.is_file():
            for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows
    settled=[e for e in journal(log.usage_path) if e.get('event') != 'ledger_initialized']
    events=journal(log.events_path)
    dispatched = sum(e.get('kind') == 'llm_request' for e in events)
    calls = max(dispatched,len(settled))
    known = sum(e.get('course_cost_rmb') or 0 for e in settled)
    missing = max(0,calls-len(settled)) + sum(e.get('course_cost_rmb') is None for e in settled)
    selected['partial_reason'] = reason
    log.write_json('final_selection.json',selected)
    report = dict(status='partial',delivery_status='partial',model_available=True,requested_planner_completed=False,
        contract_unresolved=selected.get('contract_unresolved',False),
        error=reason,method=selected['family'],source_trial=selected.get('source_trial'),refit=selected.get('refit',False),
        training_rows=selected['training_rows'],full_training_rows=selected['full_training_rows'],
        attempted_classifier_fits=attempted,completed_classifier_fits=completed,unknown_failed_attempts=unknown,
        new_api_requests=calls,known_course_cost_rmb=known,unresolved_api_requests=missing,
        complete_run_course_cost_rmb=None if missing else known,formal_test_accessed=False,
        recovery_classifier_fits=0,recovery_api_requests=0)
    log.write_json('run_outcome.json',report)
    return True
