"""Scoped assertion issues: proposal correction and execution confirmation stay separate.

Two resolution channels exist. A *knowledge* issue is a misstatement about a
proposition the executor has already bound: the trial's task kind, a recorded C
value, a measured number. Checking a corrected restatement of the same
proposition is enough to close it, and requiring a new fit for it spends a
training slot on something no fit can teach.

An *execution* issue is about an object or effect that only exists after a run
(the fitted estimator of a candidate nobody has trained yet). Such an issue
stays pending until matching execution evidence arrives.

A *measurement* issue asserts improvement, robustness or causation. Neither a
restatement nor an unrelated successful trial may close it.

`layering='off'` keeps the pre-A148 behaviour byte for byte: every issue waits
for execution evidence.
"""
from copy import deepcopy
from .fact_registry import check_claim,digest

KNOWLEDGE='knowledge';EXECUTION='execution';MEASUREMENT='measurement'
CHANNEL={KNOWLEDGE:'registry_check',EXECUTION:'execution_evidence',MEASUREMENT:'measurement_evidence'}
# Claim forms that only exist once a candidate has actually run. No producer
# emits them yet; they are reserved so a future forward-looking claim cannot
# silently inherit the cheap knowledge closure.
EXECUTION_CLAIM_KINDS=('capability','object_fit')
MEASUREMENT_CLAIM_KINDS=('uplift','robustness','causal')


def signature(claim):
    return {k:v for k,v in claim.items() if k not in ('value','direction','op','namespace','expected_role')}


def claim_class(claim,registry,report=None):
    """Which channel can close a contradiction of this claim.

    Every form the current protocol emits references a fact that already
    exists, so every one of them is knowledge: restating it correctly is
    checkable without a new run. The reserved kinds are the ones that would
    genuinely need a run or a new measurement.
    """
    if not isinstance(claim,dict):return MEASUREMENT
    kind=claim.get('kind')
    if kind in EXECUTION_CLAIM_KINDS:return EXECUTION
    if kind in MEASUREMENT_CLAIM_KINDS:return MEASUREMENT
    if kind in ('value','compare','parameter_relation'):return KNOWLEDGE
    return MEASUREMENT


def issue_for(claim,registry,trial=None,layering='off'):
    report=check_claim(claim,registry)
    if report['status']!='contradicted':return None
    issue=dict(issue_id='issue:'+digest([registry['namespace'],claim])[:20],namespace=registry['namespace'],
               code='bound_assertion_contradiction',severity='semantic',source_trial=trial,
               claim=deepcopy(claim),verification=report,signature=signature(claim),status='open',
               required_postcondition='A confirmed assertion with the same bound proposition, then matching execution evidence')
    if layering!='off':
        kind=claim_class(claim,registry,report)
        issue['issue_class']=kind;issue['resolution_channel']=CHANNEL[kind]
        issue['required_postcondition']=(
            'A confirmed assertion about the same bound proposition' if kind==KNOWLEDGE else
            'Matching execution evidence for the object or effect under test' if kind==EXECUTION else
            'New measurement evidence for the claimed improvement, robustness or cause')
        issue['new_fit_required']=kind!=KNOWLEDGE
    return issue


def historical_issues(state,registry,layering='off'):
    result=[]
    for row in state['history']:
        packet=(row.get('proposal') or {}).get('grounding') or {}
        if packet.get('namespace')!=registry['namespace']:continue
        for claim in packet.get('assertions',[]):
            issue=issue_for(claim,registry,row['trial'],layering)
            if issue is not None:result.append(issue)
    return result


def _closable_by_check(issue,correction,registry):
    return (issue.get('resolution_channel')==CHANNEL[KNOWLEDGE]
            and correction is not None and check_claim(correction,registry)['status']=='confirmed')


def address_issues(issues,assertions,addresses,registry,layering='off'):
    known={i['issue_id'] for i in issues};errors=[];result=[]
    if (not isinstance(addresses,list) or any(not isinstance(a,str) for a in addresses)
            or len(set(addresses))!=len(addresses) or not set(addresses)<=known):
        return deepcopy(issues),['Unknown or duplicate issue reference']
    for original in issues:
        issue=deepcopy(original)
        if issue['namespace']!=registry['namespace']:
            issue['status']='superseded';result.append(issue);continue
        if issue['status'] in ('resolved','resolved_semantic','superseded'):result.append(issue);continue
        corrections=[c for c in assertions if signature(c)==issue['signature'] and check_claim(c,registry)['status']=='confirmed']
        if issue['issue_id'] in addresses and corrections:
            if layering!='off' and _closable_by_check(issue,corrections[0],registry):
                issue['status']='resolved_semantic';issue['closed_by']=CHANNEL[KNOWLEDGE]
                issue['correction']=deepcopy(corrections[0]);issue['execution_confirmation']=None
            else:
                issue['status']='addressed_pending_check';issue['correction']=deepcopy(corrections[0])
        else:
            issue['status']='open';errors.append('Issue requires a checked correction: '+issue['issue_id'])
        result.append(issue)
    return result,errors


def confirm_execution(issues,row,layering='off'):
    output=deepcopy(issues)
    matched=(row.get('status')=='completed' and (row.get('metric_identity') or {}).get('status')=='confirmed'
             and (row.get('configuration_fact_check') or {}).get('status')=='confirmed')
    for issue in output:
        if issue['status']=='addressed_pending_check':
            if layering!='off' and issue.get('resolution_channel')!=CHANNEL[EXECUTION]:
                # Knowledge was already closable by a check; measurement needs its
                # own evidence and is not closed by an unrelated completed trial.
                if issue.get('resolution_channel')==CHANNEL[KNOWLEDGE]:
                    issue['status']='resolved_semantic';issue['closed_by']=CHANNEL[KNOWLEDGE]
                    issue['execution_confirmation']=None
                continue
            issue['status']='resolved' if matched else 'open'
            issue['execution_trial']=row.get('trial');issue['execution_confirmation']=matched
    return output
