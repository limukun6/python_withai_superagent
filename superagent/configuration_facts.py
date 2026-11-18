"""Configuration changes computed from declared candidates, never free-form prose."""
from copy import deepcopy
import math
from .metrics import score_of,completed


def _finite(value):return type(value) in (int,float) and math.isfinite(value)
def _id(row):return row.get('id',row.get('candidate'))


def _route(row):
    from .evidence_agent import route_of
    try:return route_of(_id(row))
    except ValueError:return None


def choose_reference(history,candidate,explicit_trial=None):
    if explicit_trial is not None:
        if type(explicit_trial) is not int:raise ValueError('Configuration reference must be an integer trial ID')
        matches=[row for row in history if row['trial']==explicit_trial]
        if len(matches)!=1:raise ValueError('Configuration reference is not an available unique trial')
        return matches[0],'structured_expectation_reference'
    eligible=[row for row in history if completed(row)]
    same=[row for row in eligible if _route(candidate) is not None and _route(row)==_route(candidate)]
    if same:return min(same,key=lambda r:(-score_of(r),r['trial'])),'best_completed_in_selected_route'
    if eligible:return min(eligible,key=lambda r:(-score_of(r),r['trial'])),'best_completed_overall'
    if history:return history[-1],'last_attempt_configuration_only'
    return None,'no_previous_trial'


def _effect(field,before,after,same_route):
    if not (_finite(before) and _finite(after)):return None
    direction='increased' if after>before else 'decreased'
    if field=='C' and min(before,after)>0:
        effect='stronger_regularization' if after<before else 'weaker_regularization'
        meaning='C is inverse regularization strength; this does not predict accuracy or gap changes'
    elif field=='l2_leaf_reg' and min(before,after)>=0:
        effect='stronger_l2_penalty' if after>before else 'weaker_l2_penalty';meaning='Direction of the declared leaf-weight penalty only'
    elif field=='min_samples_leaf' and min(before,after)>0:
        effect='larger_minimum_leaf' if after>before else 'smaller_minimum_leaf';meaning='A larger minimum leaf size restricts available tree splits'
    elif field=='depth' and min(before,after)>0:
        effect='higher_depth_cap' if after>before else 'lower_depth_cap';meaning='A lower depth cap restricts the permitted tree depth'
    else:return {'numeric_direction':direction,'declared_effect':None,'accuracy_effect_verified':False}
    return {'numeric_direction':direction,'declared_effect':effect,'definition':meaning,
        'effect_scope':'definition within a fixed parameterization',
        'same_declared_route':same_route,'accuracy_effect_verified':False,
        'cross_route_note':None if same_route else 'Different representations/algorithms can change scaling; numerical C values do not establish equal effective regularization'}


def configuration_change(reference,candidate,reference_rule):
    if reference is None:return {'available':False,'source':'executor','reason':'No previous configuration','reference_rule':reference_rule}
    if not isinstance(reference.get('params'),dict) or not isinstance(candidate.get('params'),dict):
        return {'available':False,'source':'executor','reason':'Declared parameters unavailable','reference_rule':reference_rule}
    before,after=reference['params'],candidate['params'];old_route,new_route=_route(reference),_route(candidate)
    same=old_route is not None and old_route==new_route
    changed=[]
    for key in sorted(set(before)|set(after)):
        if key in before and key in after and before[key]==after[key]:continue
        changed.append({'parameter':key,'before_present':key in before,'after_present':key in after,
            'before':deepcopy(before.get(key)),'after':deepcopy(after.get(key)),
            'parameter_effect':_effect(key,before.get(key),after.get(key),same)})
    return {'available':True,'source':'executor','reference_trial':reference['trial'],'reference_candidate':_id(reference),
        'reference_status':reference.get('status'),'reference_rule':reference_rule,'selected_candidate':_id(candidate),
        'selected_status':candidate.get('status'),
        'reference_route':old_route,'selected_route':new_route,'same_declared_route':same,
        'changed_parameters':changed,'single_declared_parameter_change':same and len(changed)==1,
        'comparison_scope':'configuration metadata only; split/training controls are separate evidence',
        'single_factor_attribution_supported':False,'statistical_test_performed':False,
        'notice':'Route or multiple fields changed; do not attribute a score difference to just one field' if not same or len(changed)>1
                 else 'Parameter direction is known; whether accuracy or generalization improves requires measurement'}


def proposal_change(proposal,history,candidate):
    if candidate is None:return {'available':False,'source':'executor','reason':'Selected candidate metadata not supplied'}
    expected=proposal.get('expected_change') or {}
    reference,rule=choose_reference(history,candidate,expected.get('reference_trial'))
    return configuration_change(reference,candidate,rule)


def latest_configuration_change(history):
    from .evidence_agent import latest_result
    result=latest_result(history);ref=result['best_prior_trial']
    if not history or ref is None:return {'available':False,'source':'executor','reason':'No earlier completed comparison'}
    reference=next(row for row in history if row['trial']==ref)
    return configuration_change(reference,history[-1],'same_best_prior_trial_as_latest_result')
