"""One predeclared real checked-history closed loop vs an existing same-source raw reference."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from scripts.campus_direct import CAMPUS_IP
from scripts.status_a29_choice_matrix import completion_failure
from superagent.ab_run import sources
from superagent.ab_flash import preflight
from superagent.cli import build_parser


ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-study','ledger','api-key-file','window-config','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();directory=root/'preference_E'
    save(root/'protocol.json',dict(batch='A58-feedback-closed-loop',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A58_feedback_closed_loop_pilot_plan.md'),maximum_new_classifier_fits=11,maximum_new_requests=3,
        reference_run_reused=True,exploratory_followup=True,automatic_retry=False,course_budget_compliance='unverified_history'))
    try:
        deadline=datetime.fromisoformat(json.loads(a.window_config.read_text(encoding='utf-8-sig'))['deadline'])
        if (deadline-datetime.now(timezone.utc)).total_seconds()<1080:raise ValueError('Less than 18 minutes; no new run started')
        if read(a.reference_study/'completion.json')['status']!='completed' or read(a.reference_study/'protocol.json')['source_files']!=frozen:
            raise ValueError('Completed same-source reference required')
        reference=a.reference_study/'preference_11';entry=next(e for e in read(a.reference_study/'protocol.json')['entries'] if e['task']=='preference')
        if sha(Path(entry['training_archive']))!=entry['training_archive_sha256']:raise ValueError('Training package changed')
        argv=['run',entry['training_archive'],'--run-dir',str(directory),'--planner','checked-history','--scheduler','ab-v1',
            '--portfolio','ab-v1','--fit-limit','11','--seed','707','--time-limit','600','--trial-timeout','60',
            '--task-profile','train','--proposal-contract','calibrated-v2','--partial-delivery','checkpoint',
            '--api-key-file',str(a.api_key_file),'--budget-ledger',str(a.ledger),'--task-budget-id',entry['task_budget_id'],
            '--llm-budget-rmb','.30','--history-policy','incremental-research']
        ledger,_,_=preflight(build_parser().parse_args(argv));protocol=read(root/'protocol.json')
        protocol.update(project_usage_before=ledger.project_summary(),reference_outcome_sha256=sha(reference/'run_outcome.json'))
        save(root/'protocol.json',protocol)
        observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),'PYTHONIOENCODING':'utf-8','SUPERAGENT_OFFLINE':'0',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','AB_A29_RUN_NAME':'preference_E'}
        code,seconds=launch([sys.executable,'-m','scripts.campus_direct','--ip',CAMPUS_IP,'superagent',*argv],root/'train.log',env,660)
        outcome=read(directory/'run_outcome.json')
        if code or not outcome['requested_planner_completed']:raise RuntimeError('No retry: '+str(completion_failure(outcome)))
        counts=outcome['classifier_fit_accounting']
        if len(physical(root))!=counts['attempted_classifier_fits'] or not 2<=counts['completed_classifier_fits']<=11 or outcome['new_api_requests']>3:
            raise ValueError('Physical fit/request accounting mismatch')
        manifest=read(directory/'run_manifest.json');old_manifest=read(reference/'run_manifest.json')
        if manifest['source_files']!=frozen or manifest['capabilities']!=old_manifest['capabilities']:raise ValueError('Source/menu mismatch')
        for split in entry['splits']:
            if read(directory/(split+'_split.json'))!=read(reference/(split+'_split.json')):raise ValueError('Split mismatch')
        def first_payload(run):
            events=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            return json.loads(next(e for e in events if e['kind']=='llm_request')['messages'][1]['content'])
        if {k:v for k,v in first_payload(directory).items() if k not in ('checks','memory')}!=first_payload(reference):
            raise ValueError('First common context differs')
        selection=read(directory/'final_selection.json')
        if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model hash differs')
        save(root/'model_frozen.json',selection);folder=root/'evaluation';folder.mkdir()
        for name in ('features.zip','holdout.joblib'):shutil.copy2(a.reference_study/'evaluation/preference'/name,folder/name)
        prediction=directory/'predictions/rows.csv'
        commands=[
            [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(prediction)],
            [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                '--holdout',str(folder/'holdout.joblib'),'--predictions',str(prediction),'--output',str(directory/'verification.json')]]
        for index,cmd in enumerate(commands):
            status,_=launch(cmd,root/f'export_{index}.log',{**env,'SUPERAGENT_OFFLINE':'1'},60)
            if status:raise RuntimeError('New checked model export failed')
        evaluation=read(directory/'verification.json');old=read(reference/'verification.json')
        def predictions(path):
            with path.open(encoding='utf-8',newline='') as file:return [r['prediction'] for r in csv.DictReader(file)]
        newp=predictions(prediction);oldp=predictions(reference/'predictions/rows.csv')
        usage=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()];calls=[r for r in usage if r['event']=='llm_call']
        result=dict(R_reference_score=old['score'],E_new_score=evaluation['score'],E_minus_R=evaluation['score']-old['score'],
            metric=evaluation['primary_metric'],rows=evaluation['rows'],prediction_disagreements=sum(x!=y for x,y in zip(newp,oldp,strict=True)),
            R_selected=read(reference/'run_outcome.json')['selected_candidate'],E_selected=outcome['selected_candidate'],first_common_context_identical=True,
            raw_reference_reused=True,independent_R_repeat_performed=False,new_checked_outcome=outcome)
        save(root/'results.json',result)
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=len(physical(root)),real_api_requests=len(calls),
            input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),
            known_new_course_cost_rmb=outcome['known_course_cost_rmb'],project_usage_after=ledger.project_summary(),
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,formal_test_opened=False,
            course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
        print(json.dumps(result))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,observed_fit_entries=len(physical(root)),elapsed_seconds=time.monotonic()-start));raise
