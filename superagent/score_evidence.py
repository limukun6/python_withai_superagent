"""Bound probability evidence for trusted local primary-metric verification."""
import hashlib
import json

import numpy as np
from .prediction_evidence import fingerprint


def matrix_sha256(values):
    matrix=np.asarray(values,dtype=float)
    return hashlib.sha256(json.dumps(matrix.tolist(),separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_score_record(prediction_record,scores,classes,metric,candidate_id):
    value=dict(schema_version=1,candidate_id=candidate_id,metric_spec=metric.record(),
        scores=np.asarray(scores,dtype=float).tolist(),class_order=np.asarray(classes).tolist(),
        scores_sha256=matrix_sha256(scores),class_order_sha256=fingerprint(classes),
        **{k:prediction_record[k] for k in ('prediction_sha256','label_sha256','evaluation_payload_sha256')})
    checked_score_record(value,prediction_record,metric,candidate_id)
    return value


def checked_score_record(value,prediction_record,metric,candidate_id,labels=None):
    if not isinstance(value,dict) or value.get('schema_version')!=1 or value.get('candidate_id')!=candidate_id:
        raise ValueError('Missing or conflicting candidate score evidence')
    if metric.check_identity(value.get('metric_spec'))['status']!='confirmed':
        raise ValueError('Probability evidence has no confirmed metric identity')
    for key in ('prediction_sha256','label_sha256','evaluation_payload_sha256'):
        if value.get(key)!=prediction_record[key]:raise ValueError('Score evidence identity differs: '+key)
    matrix=np.asarray(value['scores'],dtype=float);classes=value['class_order']
    if (not isinstance(classes,list) or len(set(classes))!=len(classes) or set(classes)!=set(metric.labels)
        or matrix.shape!=(len(prediction_record['predictions']),len(classes)) or not np.isfinite(matrix).all()
        or np.any(matrix<0) or np.any(matrix>1) or not np.allclose(matrix.sum(axis=1),1,atol=1e-6)):
        raise ValueError('Invalid bound probability shape, class domain or normalized values')
    if matrix_sha256(matrix)!=value.get('scores_sha256') or fingerprint(classes)!=value.get('class_order_sha256'):
        raise ValueError('Probability matrix or class-column fingerprint differs')
    if labels is not None:
        if fingerprint(labels)!=prediction_record['label_sha256']:raise ValueError('Trusted score labels differ')
        # This checks metric semantics, including positive class and missing classes.
        metric.evaluate(labels,prediction_record['predictions'],matrix,classes)
    return matrix,classes
