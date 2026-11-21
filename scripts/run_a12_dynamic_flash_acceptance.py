"""Offline real CLI/worker/SSE/accounting integration with explicitly synthetic HTTP."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch, MagicMock

import requests

from scripts import run_b1_ab_acceptance as fixture
from scripts.a_batch_common import save
from superagent.ab_run import sources
from superagent.cli import main
from superagent.ab_flash import MODEL
from superagent.pricing import resolve_pricing
from superagent.task_budget import TaskBudgetLedger


def synthetic_ledger(root):
    """A fake imported event, used ONLY with intercepted HTTP and never a real key."""
    price=resolve_pricing(MODEL)
    cost=price.course_cost(100,20)
    audit=dict(fixture_only=True,calls=[dict(request_id='a'*64,scope='shared_diagnostics',cost_status='known',
        course_cost_units=round(cost*1e9),course_cost_rmb=cost,input_tokens=100,output_tokens=20,
        requested_model=MODEL,served_model=MODEL,pricing=price.metadata())])
    path=root/'synthetic_history.json';save(path,audit)
    return TaskBudgetLedger.initialize(root/'synthetic_ledger.sqlite',history=path,expected_sha256=fixture.sha(path.read_bytes()))


def transport(contexts, mode='ok'):
    def post(url, **kwargs):
        payload=json.loads(kwargs['json']['messages'][1]['content'])
        contexts.append(payload)
        if mode=='timeout':raise requests.Timeout('Controlled offline timeout')
        candidate=payload['available_candidates'][0]
        reference=next(r for r in payload['history'] if r['split_id']=='primary' and r['status']=='completed')
        proposal=dict(candidate=candidate['id'],task_kind=payload['task_spec']['kind'],
            configuration_claim={'classifier':payload['candidate_facts'][candidate['id']]['classifier']},
            reason='Test the next registered configuration.',review='Read the primary measurements.',
            hypothesis='Primary score may improve.',evidence_trials=[reference['trial']],
            expected_change=dict(metric='search_score',reference_trial=reference['trial'],direction='increase',min_delta=.001),
            slice_expectation=None)
        chunk=dict(id='synthetic-offline-response',model=MODEL,choices=[dict(index=0,
            delta={'content':json.dumps(proposal) if mode!='bad_json' else 'invalid JSON'},finish_reason='stop')])
        if mode!='missing_usage':chunk['usage']=dict(prompt_tokens=100,completion_tokens=20)
        response=MagicMock();response.status_code=200;response.ok=True
        response.iter_lines.return_value=[b'data: '+json.dumps(chunk).encode(),b'data: [DONE]']
        response.__enter__.return_value=response
        return response
    session=MagicMock();session.__enter__.return_value=session;session.post.side_effect=post
    return patch('superagent.experiment_llm.requests.Session',return_value=session)


def invoke(argv, log):
    with patch.object(sys,'argv',['superagent',*argv]),log.open('w',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:main()
            except SystemExit as end:return end.code
    return 0


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    fixture.offline(output)  # OS audit blocks any unexpected real network, even though CLI enables mock LLM.
    start=time.monotonic();frozen=sources();results=[]
    save(output/'protocol.json',dict(batch='A12',source_files=frozen,baseline='0418ae1',
        runner_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/A12_dynamic_flash_plan.md').read_bytes()),
        fixture_scope='anonymous synthetic numeric rows and mocked SSE; not real Flash reasoning',
        physical_fit_limit=47,mock_http_limit=15,real_api_requests=0,data_seed=932,split_seed=41,
        paid_ledger_eligible=False,fixture_ledger_must_never_be_used_for_real_requests=True))
    def record_completion(status,**extra):
        save(output/'results.json',results)
        save(output/'completion.json',dict(status=status,observed_classifier_fits=len(fixture.trace(output)),
            mock_http_requests=sum(r.get('mock_http_requests',0) for r in results),real_api_requests=0,
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,**extra))
    try:
        _,query=fixture.packages(output)
        key=output/'fixture_credential.txt';key.write_text('SYNTHETIC-OFFLINE-ONLY',encoding='utf-8')
        ledger=synthetic_ledger(output)
        pairs={}
        cases=[(s,p,'ok') for s in ('ab-fixed','ab-v1') for p in ('raw-history','checked-history')]
        cases += [('ab-v1','checked-history',m) for m in ('timeout','missing_usage','bad_json')]
        for index,(scheduler,planner,mode) in enumerate(cases):
            name=f'{scheduler}_{planner}_{mode}';directory=output/name
            contexts=[];case_start=time.monotonic();before=len(fixture.trace(output))
            active=ledger
            if mode!='ok':
                container=output/(mode+'_account');container.mkdir();active=synthetic_ledger(container)
            argv=['run',str(output/'train.zip'),'--scheduler',scheduler,'--portfolio','ab-v1','--fit-limit','11',
                '--planner',planner,'--api-key-file',str(key),'--budget-ledger',str(active.path),
                '--task-budget-id','anonymous-fixture','--llm-budget-rmb','.30','--run-dir',str(directory),
                '--seed','41','--time-limit','240','--trial-timeout','60']
            with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),transport(contexts,mode):
                code=invoke(argv,output/(name+'.log'))
            value=fixture.read(directory/'run_outcome.json')
            fits=len(fixture.trace(output))-before
            item=dict(name=name,scheduler=scheduler,planner=planner,mode=mode,exit_code=code,
                mock_http_requests=len(contexts),observed_classifier_fits=fits,outcome=value,
                elapsed_seconds=time.monotonic()-case_start)
            results.append(item);save(output/(name+'_contexts.json'),contexts)
            if value['new_api_requests']!=len(contexts) or fits!=value['classifier_fit_accounting']['attempted_classifier_fits']:
                raise ValueError('HTTP/physical training observations differ from official outcome')
            if mode=='ok':
                if code or not value['requested_planner_completed'] or not 1<=len(contexts)<=3 or fits>11:
                    raise ValueError('Success path did not finish within fit/request caps: '+name)
                shutil.copy2(output/'test.zip',directory/'test.zip')
                item['prediction']=fixture.predict_twice(directory,query)
                events=[json.loads(s) for s in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                requested_trials=[e['trial'] for e in events if e['kind']=='ab_flash_proposal']
                actions=fixture.read(directory/'actions.json')
                if any(actions[t-1]['action']!='explore' for t in requested_trials) or 1 in requested_trials:
                    raise ValueError('A confirmation/anchor/stop made a planner request')
                common=[{k:v for k,v in c.items() if k not in ('checks','memory')} for c in contexts]
                if planner=='raw-history':pairs[scheduler]=common
                elif common!=pairs[scheduler]:raise ValueError('Raw/checked shared decision contexts differ')
            else:
                if code!=2 or value['requested_planner_completed'] or len(contexts)!=1 or fits!=1:
                    raise ValueError('Failure did not stop after one request and preserved anchor fit')
                unknown=mode!='bad_json'
                if active.snapshot('anonymous-fixture')['accounting_complete']==unknown:
                    raise ValueError('Wrong persistent accounting resolution for simulated failure')
                if unknown:
                    try:active.preflight('anonymous-fixture',.1)
                    except ValueError:item['subsequent_task_call_blocked']=True
                    else:raise ValueError('Unknown failed use failed to block the next call')
                if not (directory/'trial_1.joblib').exists():raise ValueError('Anchor model was lost')
            if len(fixture.trace(output))>47 or sum(r['mock_http_requests'] for r in results)>15:
                raise ValueError('Predeclared batch resource limit exceeded')
            record_completion('running')
            print(json.dumps(dict(case=name,fits=fits,mock_http=len(contexts),status='passed')),flush=True)
        if sources()!=frozen:raise ValueError('Production source changed during acceptance')
        record_completion('passed',cases_passed=len(results),synthetic_ledger_snapshot=ledger.snapshot('anonymous-fixture'))
    except Exception as error:
        record_completion('failed',error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    run(parser.parse_args().output.resolve())
