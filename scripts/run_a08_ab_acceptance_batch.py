"""AB action gate and dispatcher fixture; no scheduler algorithm, no real fits/API."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from a_batch_common import Journal

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    state={'ab_schema_version':1,'task':{'kind':'numeric_tabular'},
        'history':[{'trial':1,'candidate':'a','params':{},'status':'completed'},
                   {'trial':2,'candidate':'failed','params':{},'status':'failed'}],
        'available_candidates':[{'id':'b','params':{}}],
        'prediction_evidence':[{'candidate_id':'a','split_id':'primary'}],
        'budget':{'fit_limit':11,'attempted_classifier_fits':2,'completed_classifier_fits':1,'remaining_fit_allowance':9,
                  'reserved_final_refit':1,'llm_request_limit':3,'used_llm_requests':0,'unknown_failed_attempts':0}}
    capabilities={k:{'kind':'numeric_tabular','params':{},'classifier_fits':1,'refit_classifier_fits':1} for k in ('a','b','failed')}
    splits={'primary':{'id':'primary','seed':707},'confirmation_1':{'id':'confirmation_1','seed':10707}}
    explore={'action':'explore','candidate_id':'b','split_id':'primary','reason':'Fixture explore','evidence_trials':[1],'estimated_classifier_fits':1}
    confirm={**explore,'action':'confirm','candidate_id':'a','split_id':'confirmation_1'}
    stop={**explore,'action':'stop','candidate_id':None,'split_id':None,'estimated_classifier_fits':0}
    return state,capabilities,splits,explore,confirm,stop


def cases():
    s,c,p,e,f,t=fixture();rows=[]
    def add(name,action,valid=False,state=None,cap=None):rows.append((name,deepcopy(state or s),deepcopy(action),deepcopy(cap or c),valid))
    for name,action in [('explore',e),('confirm',f),('stop',t)]:add(name,action,True)
    for name,action in [('unknown_candidate',{**e,'candidate_id':'absent'}),('repeated_explore',{**e,'candidate_id':'a'}),
        ('confirm_failed',{**f,'candidate_id':'failed'}),('future_evidence',{**e,'evidence_trials':[99]}),
        ('bad_split',{**f,'split_id':'outer'}),('false_cost',{**e,'estimated_classifier_fits':0}),
        ('bool_cost',{**e,'estimated_classifier_fits':True}),('stop_selects_winner',{**t,'candidate_id':'a'})]:add(name,action)
    for name,key,value in [('inconsistent_remaining','remaining_fit_allowance',8),('unknown_attempts','attempted_classifier_fits',None),
        ('unknown_failure','unknown_failed_attempts',1),('fit_cap','fit_limit',12),('llm_cap','llm_request_limit',4)]:
        changed=deepcopy(s);changed['budget'][key]=value;add(name,e,state=changed)
    changed=deepcopy(s);changed['budget'].update(attempted_classifier_fits=10,remaining_fit_allowance=1);add('refit_reserved',e,state=changed)
    changed=deepcopy(s);changed['prediction_evidence'].append({'candidate_id':'a','split_id':'confirmation_1'});add('repeated_confirmation',f,state=changed)
    for name,field,value in [('unknown_internal_fits','classifier_fits',None),('larger_refit','refit_classifier_fits',2),('modified_params','params',{'C':10})]:
        caps=deepcopy(c);caps['b'][field]=value;add(name,e,cap=caps)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args();sys.path.insert(0,str(ROOT))
    from superagent.scheduler_bridge import validate_action,dispatch_action
    from superagent.events import RunLogger
    rows=cases();j=Journal(a.output,ROOT,{'batch':'A-08','cases':[{'name':r[0],'valid':r[4]} for r in rows],
        'scope':'A boundary with deterministic executor fixture, B Scheduler absent','classifier_fit_limit':0,'validate_only':a.validate_only})
    results=[]
    if not a.validate_only:
        for name,state,action,cap,expected in rows:
            try:
                plan=validate_action(state,action,cap,fixture()[2]);result={'name':name,'passed':expected,'admitted':True,'plan':plan}
            except ValueError as error:result={'name':name,'passed':not expected,'admitted':False,'error':str(error)}
            results.append(result)
        state,cap,splits,*actions=fixture();logger=RunLogger(j.out)
        for action in actions:
            seen=[]
            def execute(plan):seen.append(plan);return {'status':'completed','classifier_fits':plan['expected_classifier_fits'],'simulated':True}
            result=dispatch_action(state,action,cap,splits,execute,logger)
            expected=state['budget']['attempted_classifier_fits']+action['estimated_classifier_fits']
            results.append({'name':'dispatch_'+action['action'],'passed':len(seen)==1 and result['budget']['attempted_classifier_fits']==expected,'result':result})
        for name,observed in [('unknown',None),('excess',2),('underreported',0)]:
            try:dispatch_action(state,actions[0],cap,splits,lambda p:{'status':'completed','classifier_fits':observed},logger)
            except ValueError:results.append({'name':'execution_'+name,'passed':True})
            else:results.append({'name':'execution_'+name,'passed':False})
        calls=[]
        def fail(plan):calls.append(plan);raise RuntimeError('Controlled executor failure')
        try:dispatch_action(state,actions[0],cap,splits,fail,logger)
        except RuntimeError:results.append({'name':'no_automatic_retry','passed':len(calls)==1})
    j.finish(results,passed=sum(r['passed'] for r in results),cases=len(results),classifier_fits=0)
    print(json.dumps({'cases':len(results),'passed':sum(r['passed'] for r in results),'real_fits':0}))


if __name__=='__main__':main()
