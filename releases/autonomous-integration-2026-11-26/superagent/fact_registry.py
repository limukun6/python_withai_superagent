"""Run-bound evidence records and bounded assertions; never certify free-form causal prose."""
from copy import deepcopy
import hashlib
import json
import math
import operator

from .comparison_scope import comparison_scope


def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()
def number(value):return type(value) in (int,float) and math.isfinite(value)
def valid_digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def build_registry(state,capabilities,binding):
    if (not isinstance(binding,dict) or not all(valid_digest(binding.get(k)) for k in ('run_sha256','source_sha256'))
            or binding.get('origin') not in ('executor_record','controlled_fixture')):
        raise ValueError('Grounded evidence requires explicit producer/run identities')
    task=deepcopy(state['task']);metric=task['metric'];facts={}
    namespace=digest({'binding':binding,'task':task,'capabilities':capabilities})
    def add(key,kind,value,role,source,known=True,**metadata):
        facts[key]=dict(id=key,kind=kind,value=deepcopy(value) if known else None,status='known' if known else 'unknown',
                        role=role,source=source,**deepcopy(metadata))
    add('task.kind','contract',task['kind'],'declared','effective_task_contract')
    add('task.metric','contract',metric['name'],'declared','effective_task_contract')
    intent=task.get('goal_resolution',{}).get('fields',{}).get('primary_metric',{})
    add('task.intent_metric','intent',metric['name'],'declared','goal_resolution',intent.get('status')=='confirmed')
    for key in ('statistical_test_performed','slice_to_global_supported'):
        records=binding.get('protocol_facts',{})
        add('protocol.'+key,'protocol',records.get(key),'observed','explicit_executor_protocol',key in records)
    available={c['id'] for c in state['available_candidates']}
    from .evidence_agent import route_of
    def metadata(name,params,family,observed):
        try:route=route_of(name)
        except ValueError:route=None
        return dict(candidate_id=name,route=route,family=family,family_observed=observed,
                    params=deepcopy(params),parameter='C',accuracy_effect_verified=False)
    for name,cap in capabilities.items():
        base='c:'+name
        add(base+'.classifier','classifier',cap['estimator'],'declared','executor_registered_capability')
        add(base+'.untried','availability',name in available,'observed','current_available_menu')
        if 'C' in cap['params']:
            add(base+'.C','parameter',cap['params']['C'],'declared','registered_configuration',
                **metadata(name,cap['params'],cap['estimator'],False))
    seen=set()
    for row in state['history']:
        trial=row['trial']
        if type(trial) is not int or trial in seen:raise ValueError('Ambiguous trial identity in fact source')
        seen.add(trial);base='t'+str(trial);completed=row['status']=='completed'
        add(base+'.status','status',row['status'],'observed','executor_trial_status')
        family=(row.get('estimator') or {}).get('family')
        observed_family=(completed and isinstance(family,str) and bool(family)
                         and (row.get('estimator') or {}).get('source')=='fitted_estimator')
        add(base+'.classifier','classifier',family,'observed','worker_fitted_estimator',observed_family)
        if 'C' in row.get('params',{}):
            add(base+'.C','parameter',row['params']['C'],'recorded_configuration','worker_recorded_configuration',
                **metadata(row['candidate'],row['params'],family,observed_family))
        identity=(row.get('metric_identity') or {}).get('status')=='confirmed' and row.get('metric_spec')==metric
        provenance=bool(row.get('split_id')) and all(valid_digest(row.get(k)) for k in
            ('evaluation_payload_sha256','label_sha256','row_ids_sha256'))
        for phase in ('train','search'):
            values=dict(row.get(phase+'_metrics') or {})
            if metric['name'] not in values:values[metric['name']]=row.get(phase+'_score')
            for name,value in values.items():
                if name in ('rows','primary_metric','score'):continue
                scope={k:deepcopy(row.get(k)) for k in ('split_id','evaluation_payload_sha256','label_sha256',
                                                      'row_ids_sha256','primary_metric','metric_spec')}
                scope.update(phase=phase,metric_name=name)
                add(base+'.'+phase+'.'+name,'measurement',value,'observed','worker_bound_measurement',
                    # Row evaluation identities bind held-out search predictions, not
                    # the separately computed training score. Never borrow those
                    # identities to certify a training measurement.
                    phase=='search' and completed and identity and provenance and number(value),scope=scope)
    return dict(registry_version=1,namespace=namespace,binding=deepcopy(binding),task_sha256=digest(task),
                catalogue_sha256=digest(capabilities),facts=facts,causal_prose_verified=False)


def wire_registry(registry):
    columns=['id','kind','role','status','value'];scopes={};rows=[]
    for key,fact in registry['facts'].items():
        row=[deepcopy(fact[c]) for c in columns];scope=fact.get('scope')
        if scope is not None:
            sid='s'+digest(scope)[:12];scopes[sid]=scope
        else:sid=None
        rows.append(row+[sid])
    return dict(namespace=registry['namespace'],columns=columns+['scope_id'],rows=rows,scopes=scopes,
                producer=registry['binding'],interpretation='Observed, declared and recorded-configuration roles are distinct. '
                'Unknown evidence and causal hypotheses are not certified. Use only IDs from these rows.')


def check_claim(claim,registry):
    result=dict(status='unknown',reason='unsupported_or_unavailable',claim=deepcopy(claim),accuracy_effect_verified=False)
    if not isinstance(claim,dict):return result
    if claim.get('namespace',registry['namespace'])!=registry['namespace']:
        return {**result,'reason':'wrong_run_task_catalogue_or_producer_namespace'}
    facts=registry['facts'];kind=claim.get('kind')
    if kind=='value':
        fact=facts.get(claim.get('ref'))
        if fact is None or fact['status']!='known':return {**result,'reason':'fact_not_known'}
        if claim.get('expected_role',fact['role'])!=fact['role']:return {**result,'reason':'epistemic_role_mismatch'}
        value=claim.get('value');actual=fact['value']
        equal=abs(value-actual)<=.0001 if number(value) and number(actual) else type(value)==type(actual) and value==actual
        return {**result,'status':'confirmed' if equal else 'contradicted','reason':'exact_bound_fact',
                'observed':deepcopy(actual),'fact_id':fact['id'],'role':fact['role']}
    if kind=='compare':
        left,right=facts.get(claim.get('left')),facts.get(claim.get('right'))
        if any(f is None or f['status']!='known' or f['kind']!='measurement' for f in (left,right)):
            return {**result,'reason':'comparison_measurement_unavailable'}
        scope=comparison_scope(left['scope'],right['scope'],primary_metric=True)
        if (not scope['comparable'] or not scope['provenance_complete'] or
            any(left['scope'].get(k)!=right['scope'].get(k) for k in ('phase','metric_name'))):
            return {**result,'reason':'comparison_scope_not_matched'}
        operations={'>':operator.gt,'<':operator.lt,'>=':operator.ge,'<=':operator.le,'==':lambda a,b:abs(a-b)<=.0001}
        op=operations.get(claim.get('op'))
        if op is None:return result
        passed=op(left['value'],right['value'])
        return {**result,'status':'confirmed' if passed else 'contradicted','reason':'matched_measurement_comparison',
                'left_value':left['value'],'right_value':right['value']}
    if kind=='parameter_relation':
        before,after=facts.get(claim.get('before')),facts.get(claim.get('after'))
        if any(f is None or f['status']!='known' or f['kind']!='parameter' or not number(f['value']) or f['value']<=0 for f in (before,after)):
            return {**result,'reason':'parameter_reference_unavailable'}
        families={'logistic','linear_svm','rbf_svm'}
        if (not before.get('family_observed') or before.get('family') not in families or before.get('family')!=after.get('family')
            or not before.get('route') or before['route']!=after.get('route')
            or any(f.get('params',{}).get('penalty','l2') in (None,'none') for f in (before,after))
            or {k:v for k,v in before['params'].items() if k!='C'}!={k:v for k,v in after['params'].items() if k!='C'}):
            return {**result,'reason':'no_same_penalized_parameterization_and_observed_reference'}
        direction='unchanged' if before['value']==after['value'] else 'weaker_regularization' if after['value']>before['value'] else 'stronger_regularization'
        from .parameter_semantics import definition
        source_definition=definition(before['family'])
        return {**result,'status':'confirmed' if claim.get('direction')==direction else 'contradicted',
                'reason':'inverse_C_between_bound_configurations','expected_direction':direction,
                'definition_source_sha256':source_definition['excerpt_sha256'],
                'scope':'Recorded/declared configuration relation only, not a verified accuracy or causal effect'}
    return result
