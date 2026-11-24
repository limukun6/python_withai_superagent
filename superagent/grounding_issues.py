"""Scoped assertion issues: proposal correction and execution confirmation stay separate."""
from copy import deepcopy
from .fact_registry import check_claim,digest


def signature(claim):
    return {k:v for k,v in claim.items() if k not in ('value','direction','op','namespace','expected_role')}


def issue_for(claim,registry,trial=None):
    report=check_claim(claim,registry)
    if report['status']!='contradicted':return None
    return dict(issue_id='issue:'+digest([registry['namespace'],claim])[:20],namespace=registry['namespace'],
                code='bound_assertion_contradiction',severity='semantic',source_trial=trial,
                claim=deepcopy(claim),verification=report,signature=signature(claim),status='open',
                required_postcondition='A confirmed assertion with the same bound proposition, then matching execution evidence')


def historical_issues(state,registry):
    result=[]
    for row in state['history']:
        packet=(row.get('proposal') or {}).get('grounding') or {}
        if packet.get('namespace')!=registry['namespace']:continue
        for claim in packet.get('assertions',[]):
            issue=issue_for(claim,registry,row['trial'])
            if issue is not None:result.append(issue)
    return result


def address_issues(issues,assertions,addresses,registry):
    known={i['issue_id'] for i in issues};errors=[];result=[]
    if (not isinstance(addresses,list) or any(not isinstance(a,str) for a in addresses)
            or len(set(addresses))!=len(addresses) or not set(addresses)<=known):
        return deepcopy(issues),['Unknown or duplicate issue reference']
    for original in issues:
        issue=deepcopy(original)
        if issue['namespace']!=registry['namespace']:
            issue['status']='superseded';result.append(issue);continue
        if issue['status'] in ('resolved','superseded'):result.append(issue);continue
        corrections=[c for c in assertions if signature(c)==issue['signature'] and check_claim(c,registry)['status']=='confirmed']
        if issue['issue_id'] in addresses and corrections:
            issue['status']='addressed_pending_check';issue['correction']=deepcopy(corrections[0])
        else:
            issue['status']='open';errors.append('Issue requires a checked correction: '+issue['issue_id'])
        result.append(issue)
    return result,errors


def confirm_execution(issues,row):
    output=deepcopy(issues)
    matched=(row.get('status')=='completed' and (row.get('metric_identity') or {}).get('status')=='confirmed'
             and (row.get('configuration_fact_check') or {}).get('status')=='confirmed')
    for issue in output:
        if issue['status']=='addressed_pending_check':
            issue['status']='resolved' if matched else 'open'
            issue['execution_trial']=row.get('trial');issue['execution_confirmation']=matched
    return output
