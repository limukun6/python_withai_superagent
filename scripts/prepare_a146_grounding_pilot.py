"""Freeze three real state_4 prefixes and preregistered controlled errors; no fits/API."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_flash import context_for
from superagent.fact_registry import build_registry,check_claim,digest
from superagent.grounded_proposal import issues_for

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
RUNS={'coupon':BASE/'runs/a135_task1_current_v2/run',
      'sms':BASE/'runs/a122_acceptance_continued/sms/run',
      'flowers':BASE/'runs/a122_acceptance_continued/flowers/run'}
ORDER={'coupon':[('natural','legacy'),('natural','grounded'),('controlled','grounded'),('controlled','legacy')],
       'sms':[('natural','grounded'),('natural','legacy'),('controlled','legacy'),('controlled','grounded')],
       'flowers':[('controlled','grounded'),('controlled','legacy'),('natural','legacy'),('natural','grounded')]}


def controlled(task,state,caps):
    value=deepcopy(state);registry=build_registry(value,caps,value['grounding_binding'])
    if task=='coupon':
        ref='t1.search.'+state['task']['metric']['name'];actual=registry['facts'][ref]['value']
        claim=dict(kind='value',ref=ref,value=actual+.1 if actual<=.9 else actual-.1)
        sentence=f'Trial 1 achieved {claim["value"]:.8f} primary search {state["task"]["metric"]["name"]}.'
    elif task=='sms':
        choices=[]
        for c in value['available_candidates']:
            candidate=dict(kind='parameter_relation',before='t1.C',after='c:'+c['id']+'.C',direction='stronger_regularization')
            report=check_claim(candidate,registry)
            if report['status']=='contradicted' and report['expected_direction']=='weaker_regularization':choices.append(candidate)
        if not choices:raise ValueError('No existing same-route untried C increase; do not invent a new candidate')
        claim=sorted(choices,key=lambda c:c['after'])[0]
        sentence='Increasing C from trial 1 to '+claim['after']+' makes regularization stronger.'
    else:
        claim=dict(kind='value',ref='task.kind',value='single_text');sentence='The current task kind is single_text.'
    if check_claim(claim,registry)['status']!='contradicted':raise ValueError('Registered injected error is not actually contradicted')
    row=value['history'][-1]
    # Historical proposal is deliberately false, while all worker measurements stay intact.
    row['proposal']=dict(candidate=row['candidate'],task_kind=state['task']['kind'],
        configuration_claim={'classifier':caps[row['candidate']]['estimator']},reason=sentence,review=sentence,
        hypothesis='This prior interpretation requires checking.',evidence_trials=[1],expected_change=None,
        slice_expectation=None,purpose='diagnose',uncertainty='The statement may be mistaken.',
        grounding=dict(namespace=registry['namespace'],evidence_refs=['task.kind'],assertions=[claim],addresses_issue_ids=[]))
    updated=build_registry(value,caps,value['grounding_binding']);issues=issues_for(value,updated)
    if len(issues)!=1 or issues[0]['status']!='open':raise ValueError('Controlled issue was not derived from raw history')
    return value,dict(claim=claim,issue=issues[0],truth_source='Existing executor records; controlled prior-proposal injection')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','freeze','retained'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A146-prepare',source_files=source,maximum_fits=0,maximum_API_requests=0,
        runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A146_flash_grounding_pilot_plan_20261124_zh.md'),
        freeze_sha256=sha(a.freeze),retained_completion_sha256=sha(a.retained/'completion.json'),order=ORDER))
    entries=[]
    try:
        if source!=read(a.freeze)['source_files'] or not read(a.retained/'completion.json').get('eligible'):
            raise ValueError('Source freeze or retained gate failed')
        for task,run in RUNS.items():
            state=read(run/'state_4.json');manifest=read(run/'run_manifest.json');caps=manifest['capabilities']
            if len(state['history'])!=3:raise ValueError('Expected preregistered three-trial prefix')
            state['grounding_binding']=dict(run_sha256=sha(run/'run_manifest.json'),source_sha256=digest(manifest['source_files']),origin='executor_record')
            state['grounding_issues']=[];ids=[c['id'] for c in state['available_candidates']]
            injected,fault=controlled(task,state,caps);folder=out/task;folder.mkdir()
            save(folder/'capabilities.json',caps);save(folder/'fault_registration.json',fault)
            entry=dict(task=task,producer_state_sha256=sha(run/'state_4.json'),producer_manifest_sha256=sha(run/'run_manifest.json'),
                capabilities_sha256=sha(folder/'capabilities.json'),states={},contexts={})
            for condition,s in (('natural',state),('controlled',injected)):
                save(folder/(condition+'_state.json'),s);entry['states'][condition]=sha(folder/(condition+'_state.json'))
                for mode in ('legacy','grounded'):
                    contract='calibrated-v2' if mode=='legacy' else 'grounded-v1'
                    value=context_for(s,caps,ids,True,contract,'compact')
                    file=folder/(condition+'_'+mode+'.json');save(file,value)
                    entry['contexts'][condition+'_'+mode]=sha(file)
            entries.append(entry)
        if source!=hashes(ROOT):raise ValueError('Production source changed')
        save(out/'frozen_inputs.json',entries);save(out/'completion.json',dict(status='prepared',eligible_for_A146=True,
            actual_classifier_fits=0,real_API_requests=0,source_unchanged=True,
            elapsed_seconds=time.monotonic()-start,frozen_inputs_sha256=sha(out/'frozen_inputs.json')))
        print(json.dumps(dict(status='prepared',tasks=len(entries),request_contexts=12,actual_classifier_fits=0,real_API_requests=0)))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
