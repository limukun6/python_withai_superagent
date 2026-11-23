"""Six new SMS whole-loop paths on the shared C1 production snapshot; no request retry."""
import argparse
import os
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import launch, physical, ARMS
from scripts.run_a34_transfer_coverage import evaluate_inputs
from superagent.ab_run import sources, registered_capabilities, freeze_partitions
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.research_budget import research_preflight
from tools.unattended_runtime import prevent_idle_sleep

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['L1', 'E0', 'R0', 'L0', 'R1', 'E1']


def main(a):
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); frozen=sources(); results=[]; ledger=None; before=None
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SUPERAGENT_OFFLINE='0')
    helpers=[Path(__file__), ROOT/'scripts/a29_worker_observer/sitecustomize.py',
        ROOT/'scripts/run_a29_choice_matrix.py',ROOT/'scripts/run_a34_transfer_coverage.py',
        ROOT/'scripts/campus_direct.py',ROOT/'scripts/a38_predict_guard.py',ROOT/'scripts/a07_verify_export.py']
    immutable={str(p.relative_to(ROOT)):sha(p) for p in helpers}
    protocol=dict(batch='A102', source_files=frozen, script_hashes=immutable,
        plan_sha256=sha(ROOT/'docs/batches/A102_sms_six_arm_C1_plan.md'),
        order=ORDER, maximum_new_classifier_fits=66, maximum_new_API_requests=12,
        reused_classifier_fits=0, starting_A_fits=1438, research_allowance_rmb=1.20,
        history_policy='incremental-research', course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False, development_holdout_previously_seen=True,
        environment=read(a.assets/'environment.json'), assets=read(a.assets/'assets.json'))
    save(root/'protocol.json',protocol)
    try:
        if frozen!=read(ROOT/'research/a89_coupon_C1/protocol.json')['source_files']:
            raise ValueError('Production differs from shared Coupon C1; use c64bee4 checkout')
        entry=read(a.reference/'protocol.json')['entries']['sms']
        training=Path(entry['training_source'])
        if sha(training)!=entry['training_source_sha256']: raise ValueError('Training package changed')
        task=load_training_task(training); spec=TaskSpec.from_task(task).record()
        if spec!=entry['task_spec_current'] or len(task.y_train)!=3302 or spec['metric']['name']!='macro_f1':
            raise ValueError('SMS task contract/input size changed')
        caps=registered_capabilities(spec,len(task.y_train),'full')
        if len(caps)!=16: raise ValueError('Expected same executable full16 menu')
        inputs=root/'inputs'; inputs.mkdir()
        _, splits=freeze_partitions(task,inputs,707,MetricSpec.for_task(task))
        if set(splits)!={'primary','confirmation_1','confirmation_2'}: raise ValueError('Expected three partitions')
        split_registration={}
        for split in splits:
            expected=read(a.split_reference/(split+'_split.json'))
            actual=read(inputs/(split+'_split.json'))
            if actual!=expected: raise ValueError('Current-environment registered partition differs')
            split_registration[split]=dict(file_sha256=sha(a.split_reference/(split+'_split.json')),
                registration=expected,old_A47_registration=entry['registered_splits'][split])
        commands={}
        for arm in ORDER:
            planner,scheduler=ARMS[arm]
            command=['run',str(training),'--run-dir',str(root/arm),'--planner',planner,
                '--scheduler',scheduler,'--portfolio','ab-v1','--catalog-scope','full',
                '--fit-limit','11','--seed','707','--time-limit','1200','--trial-timeout','300',
                '--partial-delivery','checkpoint']
            if planner!='local':
                command+=['--api-key-file',str(a.api_key_file),'--budget-ledger',str(a.ledger),
                    '--task-budget-id','sms_spam','--llm-budget-rmb','.30',
                    '--history-policy','incremental-research','--proposal-contract','calibrated-v2']
                ledger,_,_=preflight(build_parser().parse_args(command))
                research_preflight(ledger,'sms_spam',1.20)
            commands[arm]=command
        before=ledger.project_summary()
        protocol.update(entry=entry,capabilities=caps,commands=commands,project_usage_before=before,
            registered_splits=split_registration,split_scope='New c64bee4 strong-runtime partitions; not original A47 partitions')
        save(root/'protocol.json',protocol)
        if a.validate_only:
            save(root/'completion.json',dict(status='validated',actual_classifier_fits=0,real_api_requests=0,
                elapsed_seconds=time.monotonic()-start));print('Validated six new SMS paths; zero fits/API',flush=True);return
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a29_worker_observer'),str(ROOT)]),
            'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for arm in ORDER:
            if sources()!=frozen or any(sha(ROOT/p)!=v for p,v in immutable.items()): raise ValueError('Frozen code changed')
            if ledger.project_summary()['requests']!=before['requests']+sum(r['outcome']['new_api_requests'] for r in results):
                raise ValueError('Unexpected concurrent ledger writer')
            command=commands[arm];paid=ARMS[arm][0]!='local'
            if paid: preflight(build_parser().parse_args(command))
            prefix=[sys.executable,'-m','scripts.campus_direct','--ip','10.121.10.250','superagent'] if paid else [sys.executable,'-m','superagent']
            code,seconds=launch([*prefix,*command],root/(arm+'.log'),{**env,'AB_A29_RUN_NAME':arm},1260)
            directory=root/arm
            if not (directory/'run_outcome.json').exists(): raise RuntimeError('No outcome: '+arm)
            outcome=read(directory/'run_outcome.json');count=sum(r['run_name']==arm for r in physical(root))
            results.append(dict(arm=arm,run_dir=str(directory),reused=False,new_classifier_fits=count,
                reused_classifier_fits=0,outcome=outcome,process_seconds=seconds))
            save(root/'checkpoint.json',results)
            if code or not outcome['requested_planner_completed']: raise RuntimeError('Path stopped without retry: '+arm+': '+str(outcome.get('error')))
            accounting=outcome['classifier_fit_accounting'];manifest=read(directory/'run_manifest.json')
            if not 2<=count==accounting['attempted_classifier_fits']==accounting['completed_classifier_fits']<=11:
                raise ValueError('Physical fit accounting differs')
            if manifest['source_files']!=frozen or manifest['capabilities']!=caps: raise ValueError('Source/menu differs')
            if outcome['new_api_requests']>3 or (not paid and outcome['new_api_requests']): raise ValueError('Request allowance exceeded')
            for split in splits:
                if read(directory/(split+'_split.json'))!=read(inputs/(split+'_split.json')): raise ValueError('Actual split differs')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']: raise ValueError('Model identity differs')
            results[-1]['selection']=selection;save(root/'checkpoint.json',results)
            print(dict(arm=arm,status='frozen',fits=count,requests=outcome['new_api_requests'],seconds=seconds),flush=True)
        save(root/'all_models_frozen.json',results)
        evaluation=evaluate_inputs(entry,root,6)
        for row in results:
            arm=row['arm'];directory=Path(row['run_dir']);destination=root/'predictions'/arm/'rows.csv'
            commands=[[sys.executable,'-m','scripts.a38_predict_guard',str(evaluation/'features.zip'),
                '--run-dir',str(directory),'--output',str(destination)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),
                 '--run-dir',str(directory),'--holdout',str(evaluation/'holdout.joblib'),
                 '--predictions',str(destination),'--output',str(root/(arm+'_verification.json'))]]
            for i,command in enumerate(commands):
                code,_=launch(command,root/(arm+f'_export{i}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},300)
                if code: raise RuntimeError('Post-freeze prediction failed: '+arm)
            row['evaluation']=read(root/(arm+'_verification.json'));save(root/'results.json',results)
        after=ledger.project_summary();requests=sum(r['outcome']['new_api_requests'] for r in results)
        if len(physical(root))!=sum(r['new_classifier_fits'] for r in results): raise ValueError('Unexpected inference fit')
        if after['requests']!=before['requests']+requests or after['unresolved_requests']!=before['unresolved_requests'] or after['pending_requests']:
            raise ValueError('Unexpected ledger change or unresolved request')
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=len(physical(root)),reused_classifier_fits=0,
            A_cumulative_fits_after=1438+len(physical(root)),real_api_requests=requests,
            elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==frozen,project_usage_after=after,
            course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False,formal_test_or_H_opened=False))
    except Exception as e:
        after=ledger.project_summary() if ledger else None
        save(root/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1438+len(physical(root)),real_api_requests=after['requests']-before['requests'] if before and after else 0,
            elapsed_seconds=time.monotonic()-start,automatic_retry=False,project_usage_after=after,formal_test_or_H_opened=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','split-reference','assets','ledger','api-key-file','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if a.validate_only: main(a)
    else:
        with prevent_idle_sleep(): main(a)
