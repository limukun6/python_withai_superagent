"""Current-source transfer/budget coverage study with serial fits and delayed evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import launch, physical, evaluation as course_evaluation
from superagent.ab_run import sources, freeze_partitions, registered_capabilities
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.research_budget import research_preflight


ROOT=Path(__file__).resolve().parents[1]


def prepare(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();entries=[]
    original={entry['task']:entry for entry in read(args.readiness/'readiness.json')}
    course={entry['task']:entry for entry in read(args.course_prepared/'readiness.json')}
    meta={entry['name']:entry for entry in read(args.generic_prepared/'tasks.json')}
    photo=read(args.photo_prepared/'registration.json')['photo_dev']
    save(root/'protocol.json',dict(batch='A34-prepare',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A34_A36_transfer_budget_plan.md'),actual_classifier_fits=0,real_api_requests=0))
    for name in ('coupon','preference','sms','photo'):
        folder=root/name;folder.mkdir();old=original[name]
        source=Path(course[name]['training_archive']) if name in course else Path(old['source_archive'])
        if name in course:
            if sha(source)!=course[name]['training_archive_sha256']:raise ValueError('Course train-only archive changed')
        elif sha(source)!=old['task_archive_sha256']:raise ValueError('Development train-only archive changed')
        shutil.copy2(source,folder/'train.zip')
        task=load_training_task(folder/'train.zip');spec=TaskSpec.from_task(task).record()
        caps=registered_capabilities(spec,len(task.y_train),'full')
        if spec!=old['task_spec'] or caps!=old['capabilities']:raise ValueError('Task/menu changed: '+name)
        metric=MetricSpec.for_task(task);_,splits=freeze_partitions(task,folder,707,metric)
        for split in splits:
            if read(folder/(split+'_split.json'))!=read(args.readiness/name/(split+'_split.json')):
                raise ValueError('Actual development split differs')
        entry=dict(task=name,task_budget_id=old['task_budget_id'],task_spec=spec,capabilities=caps,splits=splits,
            training_archive=str(folder/'train.zip'),training_archive_sha256=sha(folder/'train.zip'))
        if name in course:
            entry.update(source_archive=course[name]['source_archive'],source_archive_sha256=course[name]['source_archive_sha256'])
        else:
            hold=args.generic_prepared/'sms' if name=='sms' else args.photo_prepared/'photo_dev'
            expected=meta['sms']['files'] if name=='sms' else photo['files']
            for file,digest in expected.items():
                if sha(hold/file)!=digest:raise ValueError('Prepared external file changed: '+name+'/'+file)
            entry.update(evaluation_source=str(hold),evaluation_files_sha256={n:sha(hold/n) for n in ('features.zip','holdout.joblib')})
        entries.append(entry)
    save(root/'readiness.json',entries)
    save(root/'completion.json',dict(status='prepared',source_unchanged=sources()==frozen,actual_classifier_fits=0,real_api_requests=0))
    print(json.dumps(dict(status='prepared',tasks=[entry['task'] for entry in entries])))


def evaluate_inputs(entry,root,count):
    if entry['task'] in ('coupon','preference'):return course_evaluation(entry,root,count)
    source=Path(entry['evaluation_source']);folder=root/'evaluation'/entry['task'];folder.mkdir(parents=True)
    for name,digest in entry['evaluation_files_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('External frozen input changed')
    shutil.copy2(source/'features.zip',folder/'features.zip')
    held=joblib.load(source/'holdout.joblib')
    if entry['task']=='photo':held['y']=np.asarray(held['y']).astype(str)
    joblib.dump(held,folder/'holdout.joblib')
    save(folder/'source_access.json',dict(original_formal_test_opened=False,
        scope=f'Development holdout labels loaded after all {count} study models frozen'))
    return folder


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    names=('sms','photo') if args.task_set=='multimodal' else ('coupon','preference','sms','photo')
    order=[(name,mode) for index,name in enumerate(names) for mode in (('off','routes') if index%2==0 else ('routes','off'))]
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    runner_hash=sha(Path(__file__));observer_hash=sha(observer)
    deadline=datetime.fromisoformat(args.window_deadline) if args.window_deadline else None
    protocol=dict(batch='A34' if args.fit_limit==11 else 'A36',source_files=frozen,order=order,
        fit_limit=args.fit_limit,max_actual_classifier_fits=len(order)*args.fit_limit,max_real_requests=3*len(order),
        runner_sha256=runner_hash,observer_sha256=observer_hash,window_deadline=args.window_deadline,
        plan_sha256=sha(ROOT/'docs/batches/A34_A36_transfer_budget_plan.md'),execute_requested=args.execute,
        history_policy='incremental-research',course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False)
    save(root/'protocol.json',protocol)
    try:
        if read(args.after_matrix/'completion.json')['status']!='completed':raise ValueError('Primary study must complete first')
        if read(args.prepared/'protocol.json')['source_files']!=frozen or read(args.after_matrix/'protocol.json')['source_files']!=frozen:
            raise ValueError('Production source differs from the frozen studies')
        entries={entry['task']:entry for entry in read(args.prepared/'readiness.json')}
        commands=[];ledger=None
        for name,mode in order:
            entry=entries[name]
            if sha(Path(entry['training_archive']))!=entry['training_archive_sha256']:raise ValueError('Training archive changed')
            directory=root/(name+'_R1_'+mode)
            command=[sys.executable,'-m','superagent','run',entry['training_archive'],'--run-dir',str(directory),
                '--planner','raw-history','--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit',str(args.fit_limit),
                '--candidate-coverage',mode,'--seed','707','--time-limit','600','--trial-timeout','60',
                '--api-key-file',str(args.api_key_file),'--budget-ledger',str(args.ledger),
                '--task-budget-id',entry['task_budget_id'],'--llm-budget-rmb','.30','--history-policy','incremental-research']
            parsed=build_parser().parse_args(command[3:]);ledger,_,_=preflight(parsed)
            research_preflight(ledger,entry['task_budget_id'],.60)
            commands.append(dict(task=name,mode=mode,command=command))
        protocol.update(commands=commands,entries=[entries[name] for name in names],project_usage_before=ledger.project_summary())
        save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',actual_classifier_fits=0,real_api_requests=0));return
        env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),'SUPERAGENT_OFFLINE':'0'}
        for item in commands:
            if deadline is not None and (deadline-datetime.now(timezone.utc)).total_seconds()<660:
                raise RuntimeError('Window too short for another path; no request started')
            if sources()!=frozen or sha(Path(__file__))!=runner_hash or sha(observer)!=observer_hash:raise ValueError('Frozen code changed')
            directory=root/(item['task']+'_R1_'+item['mode']);local_env={**env,'AB_A29_RUN_NAME':directory.name}
            code,seconds=launch(item['command'],root/(directory.name+'.log'),local_env,660)
            outcome=read(directory/'run_outcome.json');results.append(dict(task=item['task'],mode=item['mode'],outcome=outcome,process_seconds=seconds))
            save(root/'checkpoint.json',results)
            observed=sum(row['run_name']==directory.name for row in physical(root));counts=outcome['classifier_fit_accounting']
            if code or not outcome['requested_planner_completed']:raise RuntimeError('Study stopped without retry: '+str(outcome.get('error')))
            if outcome['new_api_requests']>3:raise ValueError('Request cap exceeded')
            if not (2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=args.fit_limit):raise ValueError('Fit accounting differs')
            entry=entries[item['task']];manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entry['capabilities']:raise ValueError('Source/menu changed')
            for split in entry['splits']:
                if read(directory/(split+'_split.json'))!=read(args.prepared/item['task']/(split+'_split.json')):raise ValueError('Split differs')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model digest differs')
            results[-1]['selection']=selection;save(root/'checkpoint.json',results)
            print(json.dumps(dict(task=item['task'],mode=item['mode'],status='model_frozen',fits=observed,requests=outcome['new_api_requests'])),flush=True)
        save(root/'all_models_frozen.json',[dict(task=row['task'],mode=row['mode'],selection=row['selection']) for row in results])
        held={name:evaluate_inputs(entries[name],root,len(order)) for name in names}
        for row in results:
            directory=root/(row['task']+'_R1_'+row['mode']);folder=held[row['task']];prediction=directory/'predictions/rows.csv'
            exports=[[sys.executable,'-m','superagent','predict',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(prediction)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(prediction),'--output',str(directory/'verification.json')]]
            for index,cmd in enumerate(exports):
                code,_=launch(cmd,root/(directory.name+f'_export_{index}.log'),{**env,'SUPERAGENT_OFFLINE':'1'},60)
                if code:raise RuntimeError('Export/reload failed')
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        count=sum(row['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for row in results)
        if count!=len(physical(root)):raise ValueError('Physical fit total differs')
        save(root/'completion.json',dict(status='completed',paths_completed=len(results),actual_classifier_fits=count,
            real_api_requests=sum(row['outcome']['new_api_requests'] for row in results),
            known_new_course_cost_rmb=sum(row['outcome']['known_course_cost_rmb'] for row in results),
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,project_usage_after=ledger.project_summary(),
            original_formal_test_opened=False,course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root/'results.json',results);save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,
            observed_fit_entries=len(physical(root)),real_api_requests=sum(row['outcome']['new_api_requests'] for row in results),
            elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','execute'])
    parser.add_argument('--output',type=Path,required=True)
    for name in ('readiness','course-prepared','generic-prepared','photo-prepared','prepared','after-matrix','ledger','api-key-file'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--task-set',choices=['multimodal','all-four'],default='multimodal')
    parser.add_argument('--fit-limit',type=int,choices=[5,11],default=11)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--window-deadline')
    args=parser.parse_args()
    if args.phase=='execute' and (args.task_set,args.fit_limit) not in (('multimodal',11),('all-four',5)):
        parser.error('Registered studies are multimodal/11 or all-four/5')
    if args.phase=='prepare':prepare(args)
    else:run(args)
