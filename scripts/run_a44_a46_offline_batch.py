"""Five real dev tasks: profiled 3-fit runs and two-fit controlled-failure delivery; zero HTTP."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a12_dynamic_flash_acceptance import invoke
from scripts.run_a29_choice_matrix import physical
from superagent.ab_run import Executor,sources
from superagent.cli import build_parser
from superagent.data import load_training_task
from superagent.task_profile import training_facts


ROOT=Path(__file__).resolve().parents[1]


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    entries={e['task']:e for e in read(args.prepared/'readiness.json')}
    old={e['task']:e for e in read(args.readiness/'readiness.json')}
    save(root/'protocol.json',dict(batch='A44-A46-offline-CLI',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A44_A46_offline_structure_plan.md'),maximum_classifier_fits=25,real_api_requests=0,
        faults='One controlled exception before third worker; not a real API/network fault',
        tasks=['coupon','preference','sms','photo','bean'],external_labels_opened=False))
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network forbidden in A44-A46')
    sys.addaudithook(offline)
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    os.environ.update(AB_A29_OBSERVE_FITS='1',AB_A29_FIT_LOG=str(root/'physical_fits.jsonl'),
        PYTHONPATH=os.pathsep.join([str(observer.parent),str(ROOT)]),SUPERAGENT_OFFLINE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    worker=Executor.worker
    try:
        for name in ('coupon','preference','sms','photo','bean'):
            entry=entries.get(name)
            archive=Path(entry['training_archive']) if entry else Path(old[name]['source_archive'])
            expected=entry['training_archive_sha256'] if entry else old[name]['task_archive_sha256']
            if sha(archive)!=expected:raise ValueError('Training archive changed')
            task=load_training_task(archive);facts=training_facts(task)
            features=(Path(entry['evaluation_source'])/'features.zip') if entry and name in ('sms','photo') else args.existing_evaluation/name/'features.zip'
            if not features.is_file():raise ValueError('Existing feature-only package missing: '+name)
            for fault in (False,True):
                label=name+('_fault' if fault else '_complete');directory=root/label
                os.environ['AB_A29_RUN_NAME']=label;attempts=[]
                def controlled(executor,*a,**kw):
                    attempts.append(True)
                    if fault and len(attempts)==3:raise RuntimeError('Controlled A46 fault before third worker')
                    return worker(executor,*a,**kw)
                command=['run',str(archive),'--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','5' if fault else '3',
                    '--run-dir',str(directory),'--seed','707','--time-limit','300','--trial-timeout','60',
                    '--task-profile','train','--partial-delivery','checkpoint']
                before=len(physical(root));began=time.monotonic()
                with patch.object(Executor,'worker',controlled):code=invoke(command,root/(label+'.log'))
                value=read(directory/'run_outcome.json');selection=read(directory/'final_selection.json')
                count=len(physical(root))-before
                if count!=(2 if fault else 3) or count!=value['classifier_fit_accounting']['completed_classifier_fits']:
                    raise ValueError('Actual fit accounting differs: '+label)
                if value['new_api_requests'] or read(directory/'training_profile.json')!=facts:raise ValueError('Wrong profile or nonzero HTTP')
                states=list(directory.glob('state_*.json'))
                if not states or any(read(p)['training_profile']!=facts for p in states):raise ValueError('Profile not in real decision states')
                if fault:
                    if code!=2 or value['requested_planner_completed'] or value['delivery_status']!='partial' or not value.get('partial_recovery'):
                        raise ValueError('Fault must stay failed and produce explicit partial: '+label)
                    if value.get('error')!='Controlled A46 fault before third worker' or selection['refit']:
                        raise ValueError('Recovery hid the original fault or retrained')
                elif code or not selection['refit'] or value['delivery_status']!='complete':
                    raise ValueError('Happy path failed: '+label)
                exported=[]
                moved=root/(label+'_relocated');moved.mkdir()
                for file in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(directory/file,moved/file)
                for suffix,run_dir in (('original',directory),('relocated',moved)):
                    destination=root/(label+'_'+suffix+'.csv')
                    with (root/(label+'_'+suffix+'_predict.log')).open('w',encoding='utf-8') as log:
                        code=subprocess.run([sys.executable,'-m','scripts.a38_predict_guard',str(features),'--run-dir',str(run_dir),
                            '--output',str(destination)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
                    if code:raise ValueError('Guarded prediction failed: '+label)
                    exported.append(destination)
                if exported[0].read_bytes()!=exported[1].read_bytes():raise ValueError('Moved model predictions differ')
                results.append(dict(task=name,controlled_fault=fault,exit_code=2 if fault else 0,
                    classifier_fits=count,real_api_requests=0,profile_in_all_states=True,delivery_status=value['delivery_status'],
                    complete_run=value['requested_planner_completed'],refit=selection['refit'],training_rows=selection['training_rows'],
                    full_training_rows=len(task.y_train),prediction_rows=len(exported[0].read_text(encoding='utf-8').splitlines())-1,
                    relocated_new_process_prediction_identical=True,external_labels_opened=False,
                    elapsed_seconds=time.monotonic()-began,original_error=value.get('error'),
                    outcome_sha256=sha(directory/'run_outcome.json'),selection_sha256=sha(directory/'final_selection.json')))
                save(root/'results.json',results)
                print(json.dumps(results[-1]),flush=True)
                if sources()!=frozen or len(physical(root))>25:raise ValueError('Source or physical fit ceiling changed')
        save(root/'completion.json',dict(status='passed',paths=len(results),completed_paths=5,controlled_partial_paths=5,
            actual_classifier_fits=len(physical(root)),real_api_requests=0,model_relocations=10,
            source_unchanged=sources()==frozen,external_labels_opened=False,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),paths_verified=len(results),
            actual_classifier_fits=len(physical(root)),real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','readiness','existing-evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
