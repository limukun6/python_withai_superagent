"""Check transported run-time candidates on real histories and run a zero-API local entry fixture."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from superagent.ab_flash import context_for,validate_proposal
from superagent.ab_method_search import choose_with_history
from superagent.ab_run import sha,sources
from superagent.menu_view import decode_menu
from superagent.open_methods import capability


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_shared_runtime_menu_20261124_zh.md'


def read(path):
    return json.loads(path.read_text())


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def save(path,data):
    if path.exists():
        need(read(path)==data,'Immutable study file changed')
    else:
        with path.open('x') as stream:
            json.dump(data,stream,indent=2,ensure_ascii=False)


def menu_check(parent,slot):
    native=parent/(slot+'_open')/'agent'
    state=read(native/'state_2.json')
    g=read(native/'method_space_2.json')
    registry=read(native/g['file'])
    caps=read(native/'run_manifest.json')['capabilities']
    caps.update({k:capability(v) for k,v in registry['entries'].items()})
    eligible=[r['candidate'] for r in read(native/'method_choice_2.json')['candidates']]
    generated=[n for n in eligible if n in registry['entries']]
    need(generated,'No executable generated candidate in source state')
    counts={}
    for checked in (False,True):
        full=context_for(state,caps,eligible,checked,'calibrated-v2')
        compact=context_for(state,caps,eligible,checked,'calibrated-v2',menu_view='compact')
        decoded=decode_menu(compact)
        need(decoded==full,'Compact planner payload lost context')
        menu={r['id']:r for r in full['available_candidates']}
        need(set(menu)==set(eligible) and all(menu[n]['params']==caps[n]['params'] for n in eligible),'Menu changed parameters')
        counts['checked' if checked else 'raw']=len(menu)
    row=state['history'][0]
    name=generated[0]
    answer=dict(candidate=name,task_kind=state['task']['kind'],configuration_claim={'classifier':caps[name]['estimator']},
        reason='Check the supplied generated constructor.',review='One primary observation is available.',
        hypothesis='A different representation may change performance.',evidence_trials=[row['trial']],
        expected_change=None,slice_expectation=None,purpose='coverage',uncertainty='No result has been observed for this constructor.')
    proposal=validate_proposal(answer,state,caps,eligible,'calibrated-v2')
    need(proposal.candidate==name,'Unexpected fixture proposal')
    try:
        validate_proposal(dict(answer,candidate='not_registered'),state,caps,eligible,'calibrated-v2')
    except ValueError:
        pass
    else:
        raise ValueError('Unregistered synthetic proposal passed')
    state=copy.deepcopy(state)
    state['budget'].update(llm_request_limit=3,used_llm_requests=2)
    calls=[]
    def canned(ids):
        calls.append(tuple(ids))
        return name
    selected=choose_with_history(state,caps,eligible,[row],canned)
    need(selected['chosen']==name and calls==[tuple(eligible)],'Shared menu adapter changed the eligible set')
    state['budget']['used_llm_requests']=3
    at_limit=choose_with_history(state,caps,eligible,[row],canned)
    need('candidates' in at_limit and len(calls)==1,'Request cap did not route to local evidence')
    return dict(visible=counts,generated_visible=len(generated),full_equals_decoded=True,
                validated_fixture=name,synthetic_callbacks=1,live_flash_calls=0,request_cap_local=True)


def run(a):
    protocol=a.out/'protocol.json'
    if not protocol.exists():
        a.out.mkdir(parents=True,exist_ok=False)
        save(protocol,dict(source=sources(),plan=sha(PLAN),runner=sha(Path(__file__)),upstream='9779944d3467b9df8379101355179cb9916d5d84',
                           fit_limit=5,real_api_limit=0,fixture_train=sha(a.matrix/'inputs/q0/train.zip'),
                           fixture_features=sha(a.matrix/'inputs/q0/features.zip')))
    p=read(protocol)
    need(sources()==p['source'] and sha(PLAN)==p['plan'] and sha(Path(__file__))==p['runner'],'Engineering study changed')
    receipts={slot:menu_check(parent,slot) for parent,slots in ((a.matrix,('q1','q2','q3')),(a.released,('r1','r2'))) for slot in slots}
    save(a.out/'menu_receipts.json',receipts)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'scripts/ab_acceptance_guard'),str(ROOT)]),AB_B1_OFFLINE='1',
             AB_B1_FIT_LOG=str(a.out/'physical.jsonl'),AB_B1_PER_WORKER_FIT_LIMIT='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    agent=a.out/'agent'
    def command(args,log):
        with log.open('x') as stream:
            subprocess.run([sys.executable,'-m','superagent',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                           timeout=660,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (agent/'run_outcome.json').exists():
        command(['run',a.matrix/'inputs/q0/train.zip','--scheduler','ab-v1','--planner','local','--portfolio','ab-v1',
                 '--method-space','runtime','--task-profile','train','--fit-limit',5,'--seed',60281,
                 '--time-limit',600,'--trial-timeout',180,'--run-dir',agent],a.out/'train.log')
    local=read(agent/'run_outcome.json')
    previous=a.matrix/'q0_open'
    actions=read(agent/'actions.json')
    need(actions==read(previous/'agent/actions.json') and local['status']=='completed' and local['source_unchanged']
         and local['full_refit_completed'] and local['new_api_requests']==0,'The native shared alias changed local decisions')
    meter=[json.loads(s) for s in (a.out/'physical.jsonl').read_text().splitlines()]
    charged=read(agent/'classifier_fits.json')
    need(len(meter)==charged['attempted_classifier_fits']==charged['completed_classifier_fits']<=5 and
         meter==read(previous/'receipt.json')['physical'],'Fit ledger or selected estimators changed')
    predictions=[]
    for i in (0,1):
        dest=a.out/f'export_{i}'/'rows.csv'
        if not dest.exists():
            command(['predict',a.matrix/'inputs/q0/features.zip','--run-dir',agent,'--output',dest],a.out/f'export_{i}.log')
        meta=read(dest.parent/'prediction_manifest.json')
        need(meta['prediction_count']==33 and not meta['test_labels_read'],'Fixture prediction check failed')
        predictions.append(sha(dest))
    need(predictions[0]==predictions[1]==sha(previous/'export_0/rows.csv'),'Shared entry prediction changed')
    save(a.out/'result.json',dict(actual_fits=len(meter),native_actions=len(actions),model=read(agent/'final_selection.json')['family'],
        exports=predictions,fixture_rows=33,real_api_calls=0,real_history_states=len(receipts),menu_receipts=sha(a.out/'menu_receipts.json'),
        protocol=sha(protocol)))
    print('Matched',len(actions),'native actions and',len(meter),'fits;',len(receipts),'real states with both planner menus; 0 API calls')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('out','matrix','released'):
        parser.add_argument('--'+key,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
