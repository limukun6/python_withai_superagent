"""Describe paired prediction changes on an identical, ordered inner partition."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from .metrics import score_of,completed,metric_of,MetricSpec


def vector(values):
    array=np.asarray(values)
    if array.ndim!=1 or not len(array):raise ValueError('Prediction evidence requires a nonempty scalar vector')
    # Reject nonfinite numerical labels/predictions through strict serialization.
    json.dumps(array.tolist(),allow_nan=False)
    return array


def fingerprint(values):
    return hashlib.sha256(json.dumps(vector(values).tolist(),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_prediction_record(predictions,labels,payload_sha256):
    predictions,labels=vector(predictions),vector(labels)
    if len(predictions)!=len(labels):raise ValueError('Prediction and label counts disagree')
    if not isinstance(payload_sha256,str) or len(payload_sha256)!=64 or any(c not in '0123456789abcdef' for c in payload_sha256):
        raise ValueError('A complete evaluation payload SHA256 is required')
    return {'schema_version':1,'evaluation_payload_sha256':payload_sha256,'rows':len(labels),
        'label_sha256':fingerprint(labels),'prediction_sha256':fingerprint(predictions),
        'predictions':predictions.copy()}


def _checked_predictions(record,labels):
    if record.get('schema_version')!=1:raise ValueError('Unsupported prediction evidence schema')
    predictions=vector(record['predictions'])
    if type(record.get('rows')) is not int or record['rows']!=len(labels) or len(predictions)!=len(labels):
        raise ValueError('Prediction evidence row count mismatch')
    if record.get('label_sha256')!=fingerprint(labels):raise ValueError('Prediction evidence labels or ordering changed')
    if record.get('prediction_sha256')!=fingerprint(predictions):raise ValueError('Stored predictions changed')
    digest=record.get('evaluation_payload_sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid evaluation payload identity')
    return predictions


def compare_predictions(current,reference,labels,current_trial,reference_trial,current_candidate,reference_candidate):
    labels=vector(labels);a=_checked_predictions(current,labels);b=_checked_predictions(reference,labels)
    if current['evaluation_payload_sha256']!=reference['evaluation_payload_sha256']:
        raise ValueError('Cannot compare predictions from different evaluation payloads')
    if any(type(v) is not int or v<1 for v in (current_trial,reference_trial)) or reference_trial>=current_trial:
        raise ValueError('Prediction comparison must reference an earlier actual trial')
    correct_a=a==labels;correct_b=b==labels;n=len(labels)
    fixed=int(np.sum(correct_a & ~correct_b));lost=int(np.sum(~correct_a & correct_b))
    return {'status':'completed','current_trial':current_trial,'reference_trial':reference_trial,
        'current_candidate':current_candidate,'reference_candidate':reference_candidate,
        'reference_rule':'highest earlier completed search accuracy; earliest trial breaks ties',
        'scope':'same ordered training-internal search partition','evaluation_payload_sha256':current['evaluation_payload_sha256'],
        'evaluated_rows':n,'current_correct':int(np.sum(correct_a)),'reference_correct':int(np.sum(correct_b)),
        'both_correct':int(np.sum(correct_a & correct_b)),'newly_correct':fixed,'newly_wrong':lost,
        'both_wrong':int(np.sum(~correct_a & ~correct_b)),
        'prediction_disagreement_rows':int(np.sum(a!=b)),'net_correct_change':fixed-lost,
        'accuracy_change':(fixed-lost)/n,'single_row_accuracy_step':1/n,
        'statistical_test_performed':False,'causal_effect_established':False,
        'interpretation':'Descriptive paired counts on reused training data; not uncertainty bounds, independent confirmation, or evidence that an ensemble will improve.'}


def reference_trial(history):
    eligible=[r for r in history if completed(r)]
    return max(eligible,key=lambda r:(score_of(r),-r['trial'])) if eligible else None


def comparison_for_trial(row,history,directory,labels,expected_payload_sha256=None,metric_spec=None):
    if row.get('status')!='completed':return {'status':'unavailable','reason':'current_trial_has_no_completed_predictions'}
    reference=reference_trial(history)
    if reference is None:return {'status':'unavailable','reason':'no_prior_completed_trial'}
    records=[]
    for item in (row,reference):
        path=Path(directory)/f'trial_{item["trial"]}.joblib'
        if not path.is_file():return {'status':'unavailable','reason':'prediction_artifact_missing','trial':item['trial']}
        record=joblib.load(path).get('prediction_record')
        if record is None:return {'status':'unavailable','reason':'legacy_artifact_without_prediction_record','trial':item['trial']}
        records.append(record)
    try:
        if expected_payload_sha256 is not None and any(r.get('evaluation_payload_sha256')!=expected_payload_sha256 for r in records):
            raise ValueError('Prediction record does not match the current frozen evaluation payload')
        result=compare_predictions(*records,labels,row['trial'],reference['trial'],row['candidate'],reference['candidate'])
        result['primary_metric']=metric_of([*history,row])
        result['reference_rule']='highest earlier completed primary score; earliest trial breaks ties'
        result['primary_score_change']=score_of(row)-score_of(reference)
        result['accuracy_change_is_primary_metric_change']=result['primary_metric']=='accuracy'
        if metric_spec is not None:
            if metric_spec.name!=result['primary_metric']:raise ValueError('Evidence uses a different primary metric')
            for item,record in zip((row,reference),records):
                measured=metric_spec.evaluate(labels,record['predictions'])
                if not math.isclose(measured['score'],score_of(item),abs_tol=1e-12,rel_tol=0):
                    raise ValueError('Stored predictions disagree with the primary score')
        elif result['primary_metric']!='accuracy':raise ValueError('Nonaccuracy evidence needs the frozen class domain')
        for item,key in ((row,'current_correct'),(reference,'reference_correct')):
            if not math.isclose(result[key]/result['evaluated_rows'],item['search_accuracy'],rel_tol=0,abs_tol=1e-12):
                raise ValueError('Stored predictions disagree with the reported search accuracy')
        return result
    except (ValueError,KeyError,TypeError) as error:
        # No numerical claims leave this function when the record is inconsistent.
        return {'status':'invalid','reason':'inconsistent_prediction_evidence','error':str(error),
            'current_trial':row['trial'],'reference_trial':reference['trial']}


def visible_comparison(history):
    return deepcopy(history[-1].get('paired_error_comparison',{'status':'unavailable','reason':'legacy_history_without_prediction_evidence'})) if history else {
        'status':'unavailable','reason':'no_trials'}
