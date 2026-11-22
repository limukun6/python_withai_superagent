"""Real Flash 2x2 facts/calibration study, with explicitly selected process-only campus DNS."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from scripts.run_a34_transfer_coverage import evaluate_inputs
from scripts.campus_direct import CAMPUS_IP
from scripts.status_a29_choice_matrix import completion_failure
from superagent.ab_run import sources,freeze_partitions,registered_capabilities
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.research_budget import research_preflight


ROOT=Path(__file__).resolve().parents[1]
FACTORS={'00':('off','strict-v1'),'10':('train','strict-v1'),'01':('off','calibrated-v2'),'11':('train','calibrated-v2')}


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    window=json.loads(args.window_config.read_text(encoding='utf-8-sig'));deadline=datetime.fromisoformat(window['deadline'])
    if read(args.models_probe/'completion.json')['status']!='passed':raise ValueError('Successful actual model-list probe required')
    names=('coupon','preference') if args.task_set=='course' else ('sms','photo')
    order=[(name,arm) for name,arms in ([('coupon',['00','11','10','01']),('preference',['01','10','11','00'])]
        if args.task_set=='course' else [('sms',['00','11']),('photo',['11','00'])]) for arm in arms]
    old={row['task']:row for row in read(args.prepared/'readiness.json')};entries={}
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    files=(Path(__file__),observer,ROOT/'scripts/campus_direct.py',ROOT/'scripts/run_a29_choice_matrix.py',ROOT/'scripts/run_a34_transfer_coverage.py')
    frozen_scripts={str(p):sha(p) for p in files}
    protocol=dict(batch='A50' if args.task_set=='course' else 'A51',source_files=frozen,script_hashes=frozen_scripts,
        plan_sha256=sha(ROOT/'docs/batches/A50_A51_live_facts_calibration_plan.md'),order=order,factors=FACTORS,
        window_deadline=window['deadline'],maximum_actual_classifier_fits=11*len(order),maximum_new_requests=3*len(order),
        explicit_process_dns=CAMPUS_IP,HTTPS_hostname_preserved=True,automatic_retries=False,
        course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False,execute_requested=args.execute)
    save(root/'protocol.json',protocol)
    try:
        commands=[];ledger=None
        for name in names:
            entry=old[name];archive=Path(entry['training_archive'])
            if sha(archive)!=entry['training_archive_sha256']:raise ValueError('Training archive changed')
            task=load_training_task(archive);spec=TaskSpec.from_task(task).record();caps=registered_capabilities(spec,len(task.y_train),'full')
            if spec!=entry['task_spec'] or caps!=entry['capabilities']:raise ValueError('Input/menu changed since training registration')
            folder=root/'inputs'/name;folder.mkdir(parents=True)
            _,splits=freeze_partitions(task,folder,707,MetricSpec.for_task(task))
            entries[name]={**entry,'splits':splits}
        for name,arm in order:
            entry=entries[name];profile,contract=FACTORS[arm];directory=root/(name+'_'+arm)
            argv=['run',entry['training_archive'],'--run-dir',str(directory),'--planner','raw-history','--scheduler','ab-v1',
                '--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11','--seed','707','--time-limit','600','--trial-timeout','60',
                '--task-profile',profile,'--proposal-contract',contract,'--partial-delivery','checkpoint',
                '--api-key-file',str(args.api_key_file),'--budget-ledger',str(args.ledger),'--task-budget-id',entry['task_budget_id'],
                '--llm-budget-rmb','.30','--history-policy','incremental-research']
            ledger,_,_=preflight(build_parser().parse_args(argv))
            research_preflight(ledger,entry['task_budget_id'],1.20 if args.task_set=='course' else .60)
            commands.append(dict(task=name,arm=arm,command=[sys.executable,'-m','scripts.campus_direct','--ip',CAMPUS_IP,'superagent',*argv]))
        protocol.update(entries=list(entries.values()),commands=commands,project_usage_before=ledger.project_summary());save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',paths=len(order),actual_classifier_fits=0,real_api_requests=0));return
        env={**os.environ,'PYTHONIOENCODING':'utf-8','SUPERAGENT_OFFLINE':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)])}
        for item in commands:
            if sources()!=frozen or any(sha(Path(p))!=h for p,h in frozen_scripts.items()):raise ValueError('Frozen source changed')
            if (deadline-datetime.now(timezone.utc)).total_seconds()<660:raise RuntimeError('No complete path window remains; stopped without request')
            name,arm=item['task'],item['arm'];directory=root/(name+'_'+arm)
            code,seconds=launch(item['command'],root/(directory.name+'.log'),{**env,'AB_A29_RUN_NAME':directory.name},660)
            value=read(directory/'run_outcome.json');results.append(dict(task=name,arm=arm,outcome=value,process_seconds=seconds));save(root/'checkpoint.json',results)
            if code or not value['requested_planner_completed']:raise RuntimeError('No retry: '+directory.name+': '+str(completion_failure(value)))
            observed=sum(row['run_name']==directory.name for row in physical(root));counts=value['classifier_fit_accounting']
            if not(2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11) or value['new_api_requests']>3:
                raise ValueError('Observed supervised fit or request count differs')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entries[name]['capabilities']:raise ValueError('Source/menu differs')
            for split in entries[name]['splits']:
                if read(directory/(split+'_split.json'))!=read(root/'inputs'/name/(split+'_split.json')):raise ValueError('Actual split differs')
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            requests=[json.loads(row['messages'][1]['content']) for row in events if row['kind']=='llm_request']
            profile,contract=FACTORS[arm]
            if any(('training_profile' in row)!=(profile=='train') for row in requests):raise ValueError('Profile treatment did not reach actual requests')
            if any((row['context_version']=='ab-history-calibrated-v2')!=(contract=='calibrated-v2') for row in requests):raise ValueError('Contract treatment did not reach requests')
            selected=read(directory/'final_selection.json')
            if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Frozen model changed')
            results[-1]['selection']=selected;save(root/'checkpoint.json',results)
            print(json.dumps(dict(task=name,arm=arm,status='model_frozen',fits=observed,requests=value['new_api_requests'],seconds=seconds)),flush=True)
        save(root/'all_models_frozen.json',[dict(task=r['task'],arm=r['arm'],selection=r['selection']) for r in results])
        held={name:evaluate_inputs(entries[name],root,len(order)) for name in names}
        for row in results:
            directory=root/(row['task']+'_'+row['arm']);folder=held[row['task']];output=directory/'predictions/rows.csv'
            commands=[
                [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(output)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                    '--holdout',str(folder/'holdout.joblib'),'--predictions',str(output),'--output',str(directory/'verification.json')]]
            for i,cmd in enumerate(commands):
                code,_=launch(cmd,root/(directory.name+f'_export_{i}.log'),{**env,'SUPERAGENT_OFFLINE':'1','AB_A29_RUN_NAME':'inference_forbidden'},60)
                if code:raise RuntimeError('Frozen model export failed: '+directory.name)
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        if len(physical(root))!=sum(r['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for r in results):raise ValueError('Unexpected fit during export')
        save(root/'completion.json',dict(status='completed',paths=len(results),actual_classifier_fits=len(physical(root)),
            real_api_requests=sum(r['outcome']['new_api_requests'] for r in results),
            known_new_course_cost_rmb=sum(r['outcome']['known_course_cost_rmb'] for r in results),project_usage_after=ledger.project_summary(),
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,formal_test_opened=False,
            course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,observed_fit_entries=len(physical(root)),
            path_outcomes=len(results),real_new_request_attempts=sum(r['outcome']['new_api_requests'] for r in results),elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','models-probe','ledger','api-key-file','window-config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task-set',choices=['course','multimodal'],default='course');p.add_argument('--execute',action='store_true')
    run(p.parse_args())
