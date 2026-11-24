"""Bind a measurement claim to one completed trial, split and named metric."""
import math


def check_measurement_claim(claim, history, tolerance=0.0001):
    """Values are fractions, tolerance 0.0001 = 0.01 percentage points.

    This verifies recorded measurements, never the cause of a score or its
    significance. Missing identities and unavailable measurements stay unknown.
    """
    required={'trial','split','metric','value'}
    if not isinstance(claim,dict) or set(claim)!=required:
        raise ValueError('Measurement claim requires trial, split, metric and value')
    if (type(claim['trial']) is not int or not isinstance(claim['split'],str) or claim['split'] not in ('train','search')
            or not isinstance(claim['metric'],str) or not claim['metric']
            or type(claim['value']) not in (int,float) or not math.isfinite(claim['value'])):
        raise ValueError('Malformed measurement identity or value')
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Tolerance must be finite and nonnegative')
    matches=[r for r in history if r.get('trial')==claim['trial']]
    result={'claim':dict(claim),'observed':None,'evidence_id':None,'status':'unverifiable'}
    if not matches:
        return {**result,'status':'contradicted','reason':'trial_not_recorded'}
    if len(matches)!=1:
        return {**result,'reason':'ambiguous_trial_identity'}
    row=matches[0]
    if row.get('status')!='completed':
        return {**result,'reason':'trial_not_completed'}
    split,metric=claim['split'],claim['metric']
    value=(row.get(split+'_metrics') or {}).get(metric)
    if value is None and metric=='accuracy':value=row.get(split+'_accuracy')
    if value is None and metric==row.get('primary_metric','accuracy'):
        value=row.get(split+'_score')
    if type(value) not in (int,float) or not math.isfinite(value):
        return {**result,'reason':'named_measurement_unavailable'}
    return {**result,'observed':value,'evidence_id':f'trial:{claim["trial"]}/{split}/{metric}',
            'status':'confirmed' if abs(value-claim['value'])<=tolerance else 'contradicted',
            'reason':'bound_measurement_comparison'}
