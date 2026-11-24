"""Formal calibrated/profiled CLI using intercepted SSE; never contacts a provider."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import MagicMock,patch

import requests

from scripts import run_b1_ab_acceptance as fixture
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a12_dynamic_flash_acceptance import synthetic_ledger,invoke
from superagent.ab_run import sources
from superagent.ab_flash import MODEL


ROOT=Path(__file__).resolve().parents[1]


def intercepted(contexts,mode):
    def post(url,**kwargs):
        context=json.loads(kwargs['json']['messages'][1]['content']);contexts.append(context)
        if mode=='timeout':raise requests.Timeout('Controlled A48 intercepted timeout; no network')
        candidate=context['available_candidates'][0]
        reference=next(r for r in context['history'] if r['split_id']=='primary')
        proposal=dict(candidate=candidate['id'],task_kind=context['task_spec']['kind'],
            configuration_claim={'classifier':context['candidate_facts'][candidate['id']]['classifier']},
            reason='Test a registered alternative.',review='Read the primary observations.',hypothesis='The effect is uncertain.',
            purpose='diagnose',uncertainty='A single search split does not establish which representation is better.',
            evidence_trials=[reference['trial']],slice_expectation=None,
            expected_change=None if len(contexts)==1 else dict(metric='score_gap',reference_trial=reference['trial'],direction='decrease',min_delta=0))
        chunk=dict(id='synthetic-offline-calibrated-'+str(len(contexts)),model=MODEL,
            choices=[dict(index=0,delta={'content':json.dumps(proposal)},finish_reason='stop')],
            usage=dict(prompt_tokens=100,completion_tokens=20))
        response=MagicMock();response.status_code=200;response.ok=True;response.__enter__.return_value=response
        response.iter_lines.return_value=[b'data: '+json.dumps(chunk).encode(),b'data: [DONE]']
        return response
    session=MagicMock();session.__enter__.return_value=session;session.post.side_effect=post
    return patch('superagent.experiment_llm.requests.Session',return_value=session)


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    real_ledger_hash=sha(args.real_ledger);fixture.offline(root)
    save(root/'protocol.json',dict(batch='A48-calibrated-mock-loop',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A48_calibrated_mock_loop_plan.md'),real_api_requests=0,
        maximum_classifier_fits=11,maximum_mock_http=7,fixture_only=True,paid_ledger_eligible=False,
        real_ledger_sha256_before=real_ledger_hash))
    try:
        fixture.packages(root);key=root/'fixture_credential.txt';key.write_text('SYNTHETIC-OFFLINE-ONLY',encoding='utf-8');common={}
        for planner,mode in (('raw-history','ok'),('checked-history','ok'),('checked-history','timeout')):
            label=planner+'_'+mode;container=root/(label+'_account');container.mkdir();ledger=synthetic_ledger(container)
            directory=root/label;contexts=[];before=len(fixture.trace(root))
            argv=['run',str(root/'train.zip'),'--run-dir',str(directory),'--scheduler','ab-v1','--portfolio','ab-v1',
                '--fit-limit','5','--planner',planner,'--task-profile','train','--proposal-contract','calibrated-v2',
                '--partial-delivery','checkpoint','--api-key-file',str(key),'--budget-ledger',str(ledger.path),
                '--task-budget-id','anonymous-fixture','--llm-budget-rmb','.30','--seed','41','--time-limit','240','--trial-timeout','60']
            with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),intercepted(contexts,mode):
                code=invoke(argv,root/(label+'.log'))
            outcome=read(directory/'run_outcome.json');count=len(fixture.trace(root))-before
            if count!=outcome['classifier_fit_accounting']['attempted_classifier_fits'] or len(contexts)!=outcome['new_api_requests']:
                raise ValueError('Actual fit or mock request accounting differs')
            if not contexts or any('training_profile' not in c or c['context_version']!='ab-history-calibrated-v2' for c in contexts):
                raise ValueError('New contracts did not reach the request payload')
            if mode=='ok':
                if code or not outcome['requested_planner_completed'] or count>5:raise ValueError('Mock success loop failed')
                history=read(directory/'history.json');first=next(r for r in history if r.get('proposal'))
                if first['hypothesis_check']['status']!='not_stated' or first['proposal']['purpose']!='diagnose':
                    raise ValueError('Unknown prediction was incorrectly marked supported')
                common[planner]=[{k:v for k,v in c.items() if k not in ('checks','memory')} for c in contexts]
            elif code!=2 or count!=1 or len(contexts)!=1 or not outcome.get('partial_recovery'):
                raise ValueError('Timeout did not stop with explicit partial delivery')
            selection=read(directory/'final_selection.json');moved=root/(label+'_moved');moved.mkdir()
            for name in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(directory/name,moved/name)
            outputs=[]
            for index,run_dir in enumerate((directory,moved)):
                destination=root/(label+f'_{index}.csv')
                with (root/(label+f'_predict_{index}.log')).open('w',encoding='utf-8') as log:
                    status=subprocess.run([sys.executable,'-m','scripts.a38_predict_guard',str(root/'test.zip'),
                        '--run-dir',str(run_dir),'--output',str(destination)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
                if status:raise ValueError('Mock loop model prediction failed')
                outputs.append(destination)
            if outputs[0].read_bytes()!=outputs[1].read_bytes():raise ValueError('Relocated predictions differ')
            results.append(dict(planner=planner,mode=mode,exit_code=code,actual_classifier_fits=count,
                mock_http_requests=len(contexts),real_api_requests=0,delivery_status=outcome['delivery_status'],
                rows=31,relocated_predictions_identical=True,uncertainty_not_misclassified=True,
                synthetic_ledger_must_never_be_imported=True))
            save(root/'results.json',results)
            if len(fixture.trace(root))>11 or sum(r['mock_http_requests'] for r in results)>7:raise ValueError('Resource ceiling exceeded')
        if common['raw-history']!=common['checked-history']:raise ValueError('Raw/checked common requests differ')
        if sources()!=frozen or sha(args.real_ledger)!=real_ledger_hash:raise ValueError('Production source or real ledger changed')
        save(root/'completion.json',dict(status='passed',cases=len(results),actual_classifier_fits=len(fixture.trace(root)),
            mock_http_requests=sum(r['mock_http_requests'] for r in results),real_api_requests=0,
            common_request_payloads_identical=True,real_ledger_unchanged=True,source_unchanged=True,
            elapsed_seconds=time.monotonic()-start,paid_ledger_eligible=False))
        print(json.dumps(read(root/'completion.json')))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(fixture.trace(root)),
            real_api_requests=0,automatic_retry=False,verified_cases=len(results)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--real-ledger',type=Path,required=True)
    run(p.parse_args())
