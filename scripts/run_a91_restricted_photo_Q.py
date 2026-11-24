"""After the C1 queue exits, run only the encoder-excluded photo full menu and reuse baseline."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import time
import joblib
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from superagent.ab_run import sources,registered_capabilities
from superagent.contracts import TaskSpec
from superagent.data import load_training_task

ROOT=Path(__file__).resolve().parents[1]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);began=time.monotonic();frozen=sources();starting=None
    save(root/'protocol.json',dict(batch='A91',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A91_restricted_photo_Q_plan.md'),waiting_for=str(a.after/'completion.json'),
        maximum_classifier_fits=11,reused_baseline_fits=10,real_api_requests=0,encoder_policy='restricted'))
    try:
        while not (a.after/'completion.json').exists():
            if time.monotonic()-began>7200:raise TimeoutError('Prior queue did not exit within registered window')
            time.sleep(5)
        previous_end=read(a.after/'completion.json');starting=previous_end['A_cumulative_fits_after']
        if sources()!=frozen:raise ValueError('Production source changed while waiting')
        os.environ.update(SUPERAGENT_MINILM_WEIGHTS='',SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS='',SUPERAGENT_OFFLINE='1')
        entry=read(a.reference/'protocol.json')['entries']['photo'];baseline=a.reference/'photo_baseline'
        if read(a.reference/'protocol.json')['source_files']!=frozen:raise ValueError('Reference model source differs')
        if sha(Path(entry['training_source']))!=entry['training_source_sha256']:raise ValueError('Training input changed')
        task=load_training_task(entry['training_source']);caps=registered_capabilities(TaskSpec.from_task(task).record(),len(task.y_train),'full')
        if len(caps)!=8 or any(c['resource']=='local_frozen_encoder_cpu' for c in caps.values()):raise ValueError('Restricted menu contains an encoder or changed')
        prior=read(baseline/'final_selection.json')
        if sha(baseline/prior['model_file'])!=prior['model_sha256']:raise ValueError('Baseline model changed')
        protocol=read(root/'protocol.json');protocol.update(starting_A_fits=starting,prior_C1_status=previous_end['status'],
            capabilities=caps,reused_baseline_run=str(baseline),baseline_model_sha256=prior['model_sha256']);save(root/'protocol.json',protocol)
        directory=root/'photo_full';env={**os.environ,'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
            'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'AB_A29_RUN_NAME':'restricted_photo_full',
            'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),'PYTHONIOENCODING':'utf-8',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        train_start=time.monotonic();code,seconds=launch([sys.executable,'-m','superagent','run',entry['training_source'],'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11','--seed','707',
            '--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint'],root/'train.log',env,1260)
        if code or not (directory/'run_outcome.json').exists():raise RuntimeError('Restricted photo path failed')
        outcome=read(directory/'run_outcome.json')
        if not outcome['requested_planner_completed']:raise RuntimeError(str(outcome.get('error')))
        count=len(physical(root));manifest=read(directory/'run_manifest.json')
        if count>11 or count!=outcome['classifier_fit_accounting']['attempted_classifier_fits'] or manifest['capabilities']!=caps:
            raise ValueError('Fit count/menu differs')
        for split in entry['registered_splits']:
            if read(directory/(split+'_split.json'))!=read(a.reference/'inputs/photo'/(split+'_split.json')):raise ValueError('Actual split differs')
        history=read(directory/'history.json')
        if any('minilm' in r['candidate'] or 'mobilenet' in r['candidate'] for r in history):raise ValueError('Restricted execution used encoder')
        selected=read(directory/'final_selection.json');save(root/'model_frozen.json',selected)
        if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Frozen model changed')
        evaluation=a.reference/'evaluation/photo';destination=root/'predictions/rows.csv'
        commands=[[sys.executable,'-m','scripts.a38_predict_guard',str(evaluation/'features.zip'),'--run-dir',str(directory),'--output',str(destination)],
            [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
             '--holdout',str(evaluation/'holdout.joblib'),'--predictions',str(destination),'--output',str(root/'verification.json')]]
        for i,command in enumerate(commands):
            code,_=launch(command,root/f'export_{i}.log',env,300)
            if code:raise RuntimeError('Restricted photo delivery failed')
        truth=[str(v) for v in joblib.load(evaluation/'holdout.joblib')['y']]
        def predictions(path):
            with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            if [int(r['row_id']) for r in rows]!=list(range(len(truth))):raise ValueError('Prediction identity differs')
            return [r['prediction'] for r in rows]
        full=predictions(destination);base=predictions(baseline/'predictions/rows.csv');score=read(root/'verification.json')
        save(root/'results.json',dict(baseline=read(baseline/'verification.json'),full=score,selected_candidate=selected['family'],
            full_minus_baseline=score['score']-read(baseline/'verification.json')['score'],candidate_counts={'baseline':7,'full':8},
            full_correct_only=sum(f==y and b!=y for f,b,y in zip(full,base,truth,strict=True)),
            baseline_correct_only=sum(b==y and f!=y for f,b,y in zip(full,base,truth,strict=True)),
            prediction_disagreements=sum(f!=b for f,b in zip(full,base,strict=True)),actual_classifier_fits=count,
            reused_baseline_fits=10,process_seconds=seconds,executed_candidates=[r['candidate'] for r in history],
            no_pretrained_encoder_used=True,development_holdout_already_seen=True))
        if len(physical(root))!=count:raise ValueError('Inference trained a classifier')
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=count,A_cumulative_fits_after=starting+count,
            real_api_requests=0,source_unchanged=sources()==frozen,active_seconds=time.monotonic()-train_start,
            total_including_wait_seconds=time.monotonic()-began,formal_test_or_H_opened=False))
        print(dict(status='completed',fits=count,score=score['score']),flush=True)
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=starting+len(physical(root)) if starting is not None else None,real_api_requests=0,
            automatic_retry=False,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','after','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
