"""A-owned AB-v1 action gate. B supplies scheduling; A owns execution facts.

This module does not choose candidates or implement an exploration algorithm.
It is an injectable boundary, pending the B scheduler's integration into the CLI.
"""
from copy import deepcopy
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def validate_action(state,action,capabilities,allowed_splits):
    """Capabilities/splits are trusted executor-owned objects, not scheduler claims."""
    if state.get('ab_schema_version')!=1:raise ValueError('Expected AB schema version 1')
    fields={'action','candidate_id','split_id','reason','evidence_trials','estimated_classifier_fits'}
    if not isinstance(action,dict) or set(action)!=fields:raise ValueError('Malformed AB action fields')
    mode=action['action']
    if mode not in ('explore','confirm','stop'):raise ValueError('Unknown scheduler action')
    if not isinstance(action['reason'],str) or not action['reason'].strip():raise ValueError('Action requires a reason')
    history=state['history'];trials=[integer(r['trial'],'trial',1) for r in history]
    if len(set(trials))!=len(trials):raise ValueError('Ambiguous history trial IDs')
    refs=action['evidence_trials']
    if not isinstance(refs,list) or any(type(v) is not int for v in refs) or len(set(refs))!=len(refs) or not set(refs)<=set(trials):
        raise ValueError('Action evidence must reference distinct observed trials')
    budget=state['budget']
    limit=integer(budget['fit_limit'],'fit limit',1)
    used=integer(budget['attempted_classifier_fits'],'attempted fits')
    completed=integer(budget['completed_classifier_fits'],'completed fits')
    remaining=integer(budget['remaining_fit_allowance'],'remaining fits')
    reserved=integer(budget['reserved_final_refit'],'reserved refit')
    request_limit=integer(budget['llm_request_limit'],'request limit')
    requests=integer(budget['used_llm_requests'],'used requests')
    if limit>11 or request_limit>12 or requests>request_limit:raise ValueError('AB experiment cap exceeded')
    if budget.get('unknown_failed_attempts',0)!=0:raise ValueError('Unknown failed fit costs must be resolved before scheduling')
    if completed>used or used>limit or remaining!=limit-used:raise ValueError('Inconsistent actual fit budget')
    if reserved>remaining:raise ValueError('No budget for reserved final refit')
    estimated=integer(action['estimated_classifier_fits'],'estimated fits')
    if mode=='stop':
        if action['candidate_id'] is not None or action['split_id'] is not None or estimated!=0:
            raise ValueError('Stop must hand selection back without specifying a winner or split')
        if not any(r.get('status')=='completed' for r in history):raise ValueError('Cannot finalize without a completed candidate')
        return {'action':deepcopy(action),'candidate':None,'expected_classifier_fits':0,'state_sha256':fingerprint(state),
                'budget_before':deepcopy(budget),'scope':'stop search; common finalization separately consumes reserved refit'}
    candidate=action['candidate_id'];split=action['split_id']
    if not isinstance(candidate,str) or not isinstance(split,str):raise ValueError('Candidate and split must be strings')
    if candidate not in capabilities:raise ValueError('Candidate has no trusted executable capability')
    capability=capabilities[candidate]
    if capability.get('kind')!=state['task']['kind']:raise ValueError('Candidate does not support this input kind')
    expected=integer(capability.get('classifier_fits'),'trusted classifier fits',1)
    refit=integer(capability.get('refit_classifier_fits'),'trusted refit fits',1)
    if reserved<refit:raise ValueError('Candidate cannot be refitted within the reserved budget')
    if estimated!=expected:raise ValueError('Scheduler cost disagrees with trusted candidate cost')
    if expected>remaining-reserved:raise ValueError('Action would consume reserved refit budget')
    if split not in allowed_splits:raise ValueError('Split is not executor-registered')
    available=state['available_candidates'];ids=[c['id'] for c in available]
    if len(set(ids))!=len(ids):raise ValueError('Ambiguous available candidate IDs')
    prior=[r for r in history if r['candidate']==candidate]
    if mode=='explore':
        if prior or candidate not in ids or split!='primary':raise ValueError('Explore requires a new available candidate on primary split')
        configuration=next(c for c in available if c['id']==candidate)
    else:
        done=[r for r in prior if r.get('status')=='completed']
        if not done or split=='primary':raise ValueError('Confirm requires a completed candidate and a new split')
        if any(r.get('split_id','primary')==split and r.get('candidate',r.get('candidate_id'))==candidate
               for r in [*history,*state.get('prediction_evidence',[])]):
            raise ValueError('Candidate already observed on this split')
        configuration={'id':candidate,'params':done[0]['params']}
    if configuration['params']!=capability['params']:raise ValueError('Candidate parameters differ from trusted capability')
    return {'action':deepcopy(action),'candidate':deepcopy(configuration),'split':deepcopy(allowed_splits[split]),
            'expected_classifier_fits':expected,'state_sha256':fingerprint(state),'budget_before':deepcopy(budget)}


def dispatch_action(state,action,capabilities,allowed_splits,execute,logger):
    """Validate, journal, execute exactly once; unknown execution costs stay unknown.

The executor must enforce the admitted component count at actual training calls.
This boundary detects returned cost mismatches; it cannot undo an overrun.
"""
    plan=validate_action(state,action,capabilities,allowed_splits)
    logger.event('scheduler_action_admitted',**plan)
    try:
        observation=execute(deepcopy(plan))
    except Exception as error:
        logger.event('scheduler_action_failed',state_sha256=plan['state_sha256'],error_type=type(error).__name__,
                     message=str(error),classifier_fits=None,automatic_retry=False)
        raise
    actual=observation.get('classifier_fits')
    if (type(actual) is not int or actual<0 or actual>plan['expected_classifier_fits'] or
            observation.get('status')=='completed' and actual!=plan['expected_classifier_fits']):
        logger.event('scheduler_action_accounting_invalid',state_sha256=plan['state_sha256'],observation=observation,
                     expected_classifier_fits=plan['expected_classifier_fits'],automatic_retry=False)
        raise ValueError('Executor returned unknown or inconsistent classifier fit count')
    budget=deepcopy(plan['budget_before'])
    budget['attempted_classifier_fits']+=actual
    budget['remaining_fit_allowance']-=actual
    if observation.get('status')=='completed':budget['completed_classifier_fits']+=actual
    logger.event('scheduler_action_observed',state_sha256=plan['state_sha256'],action=action,
                 observation=observation,budget_after=budget)
    return {'observation':deepcopy(observation),'budget':budget,'state_sha256':plan['state_sha256']}
