"""Coupon six-arm whole-loop comparison, explicit L1 reuse, one cumulative ledger, no retries."""
import argparse
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical,ARMS
from scripts.run_a34_transfer_coverage import evaluate_inputs
from superagent.ab_run import sources
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.research_budget import research_preflight

ROOT=Path(__file__).resolve().parents[1]
ORDER=['L0','E0','R1','R0','E1']


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);began=time.monotonic();frozen=sources();results=[];ledger=None
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SUPERAGENT_OFFLINE='0')
    immutable={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/a29_worker_observer/sitecustomize.py',
        ROOT/'scripts/run_a29_choice_matrix.py',ROOT/'scripts/run_a34_transfer_coverage.py',ROOT/'scripts/campus_direct.py']}
    save(root/'protocol.json',dict(batch='A89',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A89_coupon_six_arm_C1_plan.md'),script_hashes=immutable,order=ORDER,
        maximum_new_classifier_fits=55,reused_L1_fits=11,maximum_new_API_requests=12,starting_A_fits=1290,
        research_allowance_rmb=1.20,history_policy='incremental-research',course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False,reference_L1_already_scored=True,
        environment=read(a.assets/'environment.json'),assets=read(a.assets/'assets.json')))
    try:
        previous=read(a.reference/'protocol.json')
        if read(a.reference/'completion.json')['status']!='completed' or previous['source_files']!=frozen:raise ValueError('Q reference/source differs')
        entry=previous['entries']['coupon'];reused=a.reference/'coupon_full';manifest=read(reused/'run_manifest.json')
        if manifest['capabilities']!=entry['current_capabilities']['full'] or manifest['source_files']!=frozen:raise ValueError('L1 catalogue/source differs')
        if manifest['fit_limit']!=11 or manifest['time_limit_seconds']!=1200 or manifest['trial_timeout_seconds']!=300:raise ValueError('L1 budget differs')
        if sha(Path(entry['training_source']))!=entry['training_source_sha256']:raise ValueError('Training source changed')
        selection=read(reused/'final_selection.json')
        if sha(reused/selection['model_file'])!=selection['model_sha256']:raise ValueError('L1 model changed')
        results.append(dict(arm='L1',run_dir=str(reused),reused=True,new_classifier_fits=0,reused_classifier_fits=11,
            outcome=read(reused/'run_outcome.json'),selection=selection,process_seconds=None,
            reference_process_seconds=next(r['process_seconds'] for r in read(a.reference/'results.json') if r['task']=='coupon' and r['scope']=='full')))
        commands={}
        for arm in ORDER:
            planner,scheduler=ARMS[arm]
            command=['run',entry['training_source'],'--run-dir',str(root/arm),'--planner',planner,'--scheduler',scheduler,
                '--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11','--seed','707',
                '--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint']
            if planner!='local':
                command+=['--api-key-file',str(a.api_key_file),'--budget-ledger',str(a.ledger),'--task-budget-id',entry['task_budget_id'],
                    '--llm-budget-rmb','.30','--history-policy','incremental-research','--proposal-contract','calibrated-v2']
                ledger,_,_=preflight(build_parser().parse_args(command));research_preflight(ledger,entry['task_budget_id'],1.20)
            commands[arm]=command
        protocol=read(root/'protocol.json');protocol.update(entry=entry,commands=commands,project_usage_before=ledger.project_summary())
        save(root/'protocol.json',protocol);save(root/'checkpoint.json',results)
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a29_worker_observer'),str(ROOT)]),'PYTHONIOENCODING':'utf-8',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for arm in ORDER:
            if sources()!=frozen or any(sha(ROOT/p)!=digest for p,digest in immutable.items()):raise ValueError('Frozen code changed')
            command=commands[arm];paid=ARMS[arm][0]!='local'
            if paid:preflight(build_parser().parse_args(command))
            prefix=[sys.executable,'-m','scripts.campus_direct','--ip','10.121.10.250','superagent'] if paid else [sys.executable,'-m','superagent']
            code,seconds=launch([*prefix,*command],root/(arm+'.log'),{**env,'AB_A29_RUN_NAME':arm},1260)
            directory=root/arm
            if not (directory/'run_outcome.json').exists():raise RuntimeError('Official path failed before outcome: '+arm)
            outcome=read(directory/'run_outcome.json');count=sum(r['run_name']==arm for r in physical(root))
            results.append(dict(arm=arm,run_dir=str(directory),reused=False,new_classifier_fits=count,outcome=outcome,process_seconds=seconds))
            save(root/'checkpoint.json',results)
            if code or not outcome['requested_planner_completed']:raise RuntimeError('Path stopped, no retry: '+arm+': '+str(outcome.get('error')))
            accounting=outcome['classifier_fit_accounting'];active=read(directory/'run_manifest.json')
            if not 2<=count==accounting['attempted_classifier_fits']==accounting['completed_classifier_fits']<=11:raise ValueError('Fit count differs')
            if active['source_files']!=frozen or active['capabilities']!=entry['current_capabilities']['full']:raise ValueError('Source/menu differs')
            if outcome['new_api_requests']>3 or not paid and outcome['new_api_requests']!=0:raise ValueError('Request allowance differs')
            for split in entry['registered_splits']:
                if read(directory/(split+'_split.json'))!=read(a.reference/'inputs/coupon'/(split+'_split.json')):raise ValueError('Actual partition differs')
            selected=read(directory/'final_selection.json')
            if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Frozen model changed')
            results[-1]['selection']=selected;save(root/'checkpoint.json',results)
            print(dict(arm=arm,status='model_frozen',fits=count,requests=outcome['new_api_requests'],seconds=seconds),flush=True)
        save(root/'all_models_frozen.json',[dict(arm=r['arm'],run_dir=r['run_dir'],selection=r['selection'],reused=r['reused']) for r in results])
        folder=evaluate_inputs(entry,root,6)
        for row in results:
            directory=Path(row['run_dir']);destination=root/'predictions'/row['arm']/'rows.csv'
            commands=[[sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(destination)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(destination),'--output',str(root/(row['arm']+'_verification.json'))]]
            for i,command in enumerate(commands):
                code,_=launch(command,root/(row['arm']+f'_export_{i}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},300)
                if code:raise RuntimeError('Post-freeze prediction failed: '+row['arm'])
            row['evaluation']=read(root/(row['arm']+'_verification.json'));save(root/'results.json',results)
        if len(physical(root))!=sum(r['new_classifier_fits'] for r in results):raise ValueError('Unexpected fit during inference')
        save(root/'completion.json',dict(status='completed',logical_paths=6,new_physical_paths=5,reused_paths=1,
            actual_classifier_fits=len(physical(root)),reused_classifier_fits=11,A_cumulative_fits_after=1290+len(physical(root)),
            real_api_requests=sum(r['outcome']['new_api_requests'] for r in results),elapsed_seconds=time.monotonic()-began,
            source_unchanged=sources()==frozen,project_usage_after=ledger.project_summary(),formal_test_or_H_opened=False,
            course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1290+len(physical(root)),reused_classifier_fits=11,automatic_retry=False,
            elapsed_seconds=time.monotonic()-began,project_usage_after=ledger.project_summary() if ledger else None,
            completed_or_attempted_paths=len(results),formal_test_or_H_opened=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','assets','ledger','api-key-file','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
