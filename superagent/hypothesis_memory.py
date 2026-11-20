"""Rebuild descriptive hypothesis memory; never infer causal truth from a score."""
from copy import deepcopy
from .metrics import completed,score_of,metric_of


def build_memory(history,available,primary_metric=None):
    from .evidence_agent import route_board,route_of
    name=metric_of(history,primary_metric or 'accuracy')
    if primary_metric is not None and name!=primary_metric:raise ValueError('Task metric disagrees with history')
    hypotheses=[]
    for row in history:
        proposal=row.get('proposal')
        if not proposal:continue
        check=deepcopy(row.get('hypothesis_check',{'status':'not_evaluable'}))
        status=check['status']
        mapped={'supported':'measurement_supported','not_supported':'measurement_not_supported',
            'not_evaluable':'not_evaluable','not_stated':'no_numerical_expectation'}
        hypotheses.append({'id':f'h{row["trial"]}','proposed_before_trial':row['trial'],'tested_by_trial':row['trial'],
            'candidate':row['candidate'],'route':route_of(row['candidate']),'text':proposal.get('hypothesis'),
            'evidence_trials':deepcopy(proposal.get('evidence_trials',[])),
            'expected_change':deepcopy(proposal.get('expected_change')),'status':mapped[status],
            'slice_expectation':deepcopy(proposal.get('slice_expectation')),
            'measurement_check':check,'causal_explanation_verified':False,
            'pending_check':deepcopy(proposal.get('route_plan',{}).get('next_check'))})
    routes=route_board(history,available,name)
    revisit={}
    for index,row in enumerate(history):
        plan=(row.get('proposal') or {}).get('route_plan') or {}
        if index and route_of(row['candidate'])!=route_of(history[index-1]['candidate']) and plan.get('revisit_previous_if'):
            revisit[route_of(history[index-1]['candidate'])]=plan['revisit_previous_if']
    for route in routes:
        if route['route'] in revisit:route['resume_if']=revisit[route['route']]
    return {'schema_version':1,'primary_metric':name,'hypotheses':hypotheses,
        'configuration_checks':[{'trial':r['trial'],'candidate':r['candidate'],
            'source':'worker_fitted_estimator_comparison','check':deepcopy(r['configuration_fact_check'])}
            for r in history if r.get('configuration_fact_check')],
        'explanation_checks':[{'trial':r['trial'],'candidate':r['candidate'],
            'source':'bounded_executor_prose_audit','warnings':deepcopy(r['explanation_audit'].get('warnings',[])),
            'fully_verified':False}
            for r in history if r.get('explanation_audit')],
        'routes':routes,
        'failures':[{'trial':r['trial'],'candidate':r['candidate'],'status':r['status']} for r in history if r['status']!='completed'],
        'open_hypotheses':[r['id'] for r in hypotheses if r['status'] in ('not_evaluable','no_numerical_expectation')],
        'interpretation':'Checks describe predeclared numerical expectations on reused internal rows. No causal claim, automatic route deletion, or significance claim.'}
