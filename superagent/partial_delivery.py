"""Freeze an already fitted primary checkpoint after failure, never refit or rescore validation."""
import json
import hashlib
import math
from pathlib import Path
import time

import joblib

from .ab_scheduler import PRIMARY, selection_evidence
from .data import DevelopmentTask
from .finalization import finalize
from .metrics import MetricSpec
from .models import predict_with_model
from .prediction_evidence import fingerprint


def freeze_checkpoint(task,state,trusted,caps,directory,logger,reason):
    directory=Path(directory)
    if (directory/'final_selection.json').exists():
        raise ValueError('Do not replace an existing frozen final selection')
    selection_evidence(state,trusted_partitions=trusted)  # Checks score, split, label and prediction identities.
    metric=MetricSpec.for_task(task)
    rows=[r for r in state['history'] if r['status']=='completed' and r['split_id']==PRIMARY
          and r['primary_metric']==metric.name and math.isfinite(r['search_score'])]
    if not rows:raise ValueError('No verified successful primary model is available')
    chosen=sorted(rows,key=lambda r:(-r['search_score'],r['candidate']))[0]
    path=directory/f'trial_{chosen["trial"]}.joblib'
    counts=json.loads(path.with_suffix('.fits.json').read_text(encoding='utf-8'))
    expected=caps[chosen['candidate']]['classifier_fits']
    if not(counts['status']=='completed' and counts['candidate']==chosen['candidate'] and expected>=1 and
           counts['completed_classifier_fits']==counts['attempted_classifier_fits']==expected):
        raise ValueError('Checkpoint lacks completed admitted-fit evidence')
    bundle=joblib.load(path);resolved=bundle['metrics']['resolved_configuration']
    if resolved['id']!=chosen['candidate'] or resolved['params']!=chosen['params']:
        raise ValueError('Checkpoint configuration differs from the successful trial')
    if metric.check_identity(bundle['metrics'].get('metric_spec'))['status']!='confirmed':
        raise ValueError('Checkpoint metric differs from the task contract')
    payload_path=directory/(PRIMARY+'.joblib')
    if hashlib.sha256(payload_path.read_bytes()).hexdigest()!=trusted[PRIMARY]['evaluation_payload_sha256']:
        raise ValueError('Trusted primary payload changed before recovery')
    payload=joblib.load(payload_path)
    if payload['kind']!=task.kind or caps[chosen['candidate']]['input_kind']!=task.kind:
        raise ValueError('Checkpoint input kind differs from the trusted payload/capability')
    predicted=predict_with_model(task.kind,bundle['model'],payload['X_search'])
    evidence=next(r for r in state['prediction_evidence'] if r['trial']==chosen['trial'])
    if fingerprint(predicted.tolist())!=evidence['prediction_sha256']:
        raise ValueError('Checkpoint no longer reproduces its verified primary predictions')
    clean=DevelopmentTask(task.kind,'partial checkpoint',task.X_train,task.y_train,None,None,task.spec)
    def no_training(*a,**kw):raise RuntimeError('Partial recovery cannot train a model')
    result=finalize(clean,rows,bundle['model'],chosen['candidate'],chosen['search_score'],payload,
        directory,logger,no_training,time.monotonic(),1,1,False,False)
    selected=json.loads((directory/'final_selection.json').read_text(encoding='utf-8'))
    selected.update(delivery_status='partial',requested_planner_completed=False,partial_reason=reason,
        full_training_rows=len(task.y_train),recovery_classifier_fits=0,recovery_api_requests=0,
        checkpoint_selection_rule='maximum verified primary metric; ties by candidate id')
    logger.write_json('final_selection.json',selected)
    logger.event('partial_delivery_frozen',candidate=chosen['candidate'],source_trial=chosen['trial'],
        checkpoint_training_rows=len(payload['y_train']),full_training_rows=len(task.y_train),
        refit_performed=False,additional_classifier_fits=0,additional_api_requests=0)
    return dict(candidate=chosen['candidate'],source_trial=chosen['trial'],delivery_status='partial',
        training_rows=len(payload['y_train']),full_training_rows=len(task.y_train),
        additional_classifier_fits=0,additional_api_requests=0,external_validation_scored=False)
