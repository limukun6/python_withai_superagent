"""Grounded proposal boundary: checked decision basis, scoped issues, unverified free prose."""
from copy import deepcopy

from .fact_registry import build_registry,check_claim,wire_registry
from .grounding_issues import historical_issues,address_issues,issue_for

INSTRUCTION=(
    ' Additionally return grounding with EXACT keys namespace, evidence_refs, assertions, addresses_issue_ids. '
    'Copy namespace from fact_registry. evidence_refs is 1..8 distinct known fact IDs and MUST include '
    'c:<chosen candidate>.classifier. For improve include a known primary search measurement of the task metric; '
    'for coverage include c:<chosen candidate>.untried; diagnose must address an open issue. '
    'assertions is 1..8 machine-checkable objects. Supported forms: '
    '{kind:value,ref:<fact ID>,value:<literal>}; {kind:compare,left:<measurement ID>,op: > or < or ==,right:<measurement ID>}; '
    '{kind:parameter_relation,before:<old C ID>,after:<new C ID>,direction:weaker_regularization or stronger_regularization or unchanged}. '
    'Numeric comparisons require matching phase, metric, class order and partition identities. '
    'C direction applies only to the same penalized parameterization and recorded/declared configurations, not accuracy. '
    'addresses_issue_ids lists each open issue you correct; restate a confirmed assertion about the same bound proposition. '
    'Do not cite unknown values as verified evidence. Put causal speculation only in hypothesis/uncertainty. '
    'Free reason/review text remains unverified; the executor compiles the authoritative basis from checked references. '
    'A correction is pending execution confirmation; do not claim the experiment already ran.')


class GroundingError(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__('Grounded proposal requires repair: '+'; '.join(report['errors']))


def registry_for(state,capabilities):
    return build_registry(state,capabilities,state.get('grounding_binding'))


def issues_for(state,registry,layering='off'):
    result={}
    for issue in historical_issues(state,registry,layering):result[issue['issue_id']]=issue
    for original in state.get('grounding_issues',[]):
        issue=deepcopy(original)
        if issue.get('namespace')!=registry['namespace']:issue['status']='superseded'
        result[issue['issue_id']]=issue
    return list(result.values())


def augment_context(base,state,capabilities,layering='off'):
    registry=registry_for(state,capabilities)
    value=deepcopy(base);value['context_version']='ab-grounded-v1';value['instruction']+=INSTRUCTION
    value['issue_layering']=layering
    if layering!='off':
        value['instruction']+=(' Issue layering is on. A knowledge correction (a wrong recorded number, '
            'a declared parameter value, the task kind, a cross-scope reference) is closed by restating '
            'the same bound proposition correctly and does not need a new experiment. Execution facts '
            'and claims of improvement still do.')
    value['fact_registry']=wire_registry(registry);value['grounding_issues']=issues_for(state,registry,layering)
    return value


def assess(answer,state,capabilities,eligible_ids,layering='off'):
    from .ab_flash import validate_proposal
    from .contracts import ExperimentProposal
    registry=registry_for(state,capabilities);issues=issues_for(state,registry,layering);errors=[];checks=[]
    def reject():
        raise GroundingError(dict(status='requires_semantic_repair',namespace=registry['namespace'],errors=errors,
             assertion_checks=checks,issues=issues,freeform_prose_verified=False))
    if not isinstance(answer,dict) or 'grounding' not in answer:
        errors.append('Missing grounding packet');reject()
    base={k:deepcopy(v) for k,v in answer.items() if k!='grounding'}
    try:validate_proposal(base,state,capabilities,eligible_ids,'calibrated-v2')
    except (ValueError,TypeError,KeyError) as e:errors.append('Execution contract: '+str(e));reject()
    packet=answer['grounding']
    if not isinstance(packet,dict) or set(packet)!={'namespace','evidence_refs','assertions','addresses_issue_ids'}:
        errors.append('Malformed grounding packet');reject()
    if packet['namespace']!=registry['namespace']:errors.append('Namespace is not the current run/task/catalogue/producer');reject()
    refs=packet['evidence_refs'];assertions=packet['assertions']
    if not isinstance(refs,list) or not 1<=len(refs)<=8 or any(not isinstance(r,str) for r in refs) or len(set(refs))!=len(refs):
        errors.append('Evidence references require 1..8 distinct IDs');reject()
    facts=registry['facts']
    if any(r not in facts or facts[r]['status']!='known' for r in refs):errors.append('Unavailable or unknown evidence reference')
    candidate=answer['candidate'];classifier='c:'+candidate+'.classifier'
    if classifier not in refs:errors.append('Selected candidate declaration must be referenced')
    if answer['purpose']=='improve' and not any(r in facts and facts[r]['kind']=='measurement' and facts[r]['status']=='known'
         and facts[r]['scope']['phase']=='search' and facts[r]['scope']['split_id']=='primary'
         and facts[r]['scope']['metric_name']==state['task']['metric']['name'] for r in refs):
        errors.append('Improve requires a bound primary task-metric measurement')
    if answer['purpose']=='coverage' and 'c:'+candidate+'.untried' not in refs:
        errors.append('Coverage must reference the untried candidate fact')
    if answer['purpose']=='diagnose' and not packet['addresses_issue_ids']:errors.append('Diagnosis must identify an open issue')
    if not isinstance(assertions,list) or not 1<=len(assertions)<=8:errors.append('Expected 1..8 checked assertions');reject()
    for claim in assertions:
        report=check_claim(claim,registry);checks.append(report)
        if report['status']!='confirmed':
            errors.append('Assertion is '+report['status']+': '+report['reason'])
            issue=issue_for(claim,registry,layering=layering)
            if issue is not None and issue['issue_id'] not in {i['issue_id'] for i in issues}:issues.append(issue)
    issues,issue_errors=address_issues(issues,assertions,packet['addresses_issue_ids'],registry,layering);errors.extend(issue_errors)
    if errors:reject()
    verified_basis=[dict(id=r,value=deepcopy(facts[r]['value']),role=facts[r]['role']) for r in refs]
    compiled='Checked '+answer['purpose']+' proposal '+candidate+' using '+', '.join(refs)+'. Future hypothesis remains unverified.'
    config=dict(id=candidate,params=deepcopy(capabilities[candidate]['params']))
    proposal=ExperimentProposal.from_decision(len(state['history'])+1,config,compiled,answer,state['history'])
    return proposal,dict(status='accepted',namespace=registry['namespace'],errors=[],assertion_checks=checks,
          verified_basis=verified_basis,issues=issues,freeform_prose_verified=False,execution_confirmation_pending=True)
