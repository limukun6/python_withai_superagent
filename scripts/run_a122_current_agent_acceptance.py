"""Three current-source autonomous local paths; fixed historical training rows and no API/test access."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
TASKS=[('coupon','a89_coupon_C1',BASE/'runs/a89_coupon_C1/evaluation/coupon/features.zip'),
       ('sms','a102_sms_C1',BASE/'runs/a102_sms_C1/evaluation/sms/features.zip'),
       ('flowers','a97_flower_C1',BASE/'runs/a95_object_photos/features.zip')]


def environment(root):
    return {**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(BASE/'runs/a82_encoder_assets/minilm'),
        'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(BASE/'runs/a82_encoder_assets/mobilenet_v3_small-047dcff4.pth'),
        'SUPERAGENT_DEEP_MINILM_WEIGHTS':str(BASE/'runs/a113R_deep_assets/minilm_l12'),
        'SUPERAGENT_AB_PREFERENCE_CATALOG':'standard',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SUPERAGENT_OFFLINE':'1',
        'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
        'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a29_worker_observer'),str(ROOT)]),
        'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}


def physical(root):
    path=root/'physical_fits.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def child(command,log,env,timeout=180,allow_failure=False):
    start=time.monotonic()
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    if process.returncode and not allow_failure:raise RuntimeError('Child failed; see '+str(log))
    return dict(returncode=process.returncode,elapsed_seconds=time.monotonic()-start)


def prepare(output):
    from scripts import a38_predict_guard
    from superagent.ab_run import freeze_partitions,registered_capabilities
    from superagent.data import load_training_task
    from superagent.contracts import TaskSpec
    from superagent.metrics import MetricSpec
    os.environ.update(environment(output));frozen=hashes(ROOT);entries=[]
    for name,previous,features in TASKS:
        prior=read(ROOT/'research'/previous/'protocol.json');command=prior['commands']['L0']
        training=Path(command[1]);oldrun=BASE/'runs'/previous/'L0'
        manifest=read(oldrun/'run_manifest.json')
        if sha(training)!=manifest['task_archive_sha256']:raise ValueError('Original training bytes changed: '+name)
        if not features.is_file():raise FileNotFoundError(features)
        folder=output/name;folder.mkdir();copied=folder/'anonymous.zip';shutil.copy2(training,copied)
        task=load_training_task(copied)
        if getattr(task,'X_val',None) is not None or getattr(task,'y_val',None) is not None:raise ValueError('Unexpected external validation')
        spec=TaskSpec.from_task(task).record();caps=registered_capabilities(spec,len(task.y_train),'full')
        if caps!=manifest['capabilities']:raise ValueError('Default catalogue changed: '+name)
        freeze_partitions(task,folder,707,MetricSpec.for_task(task))
        split_shas={}
        for split in ('primary','confirmation_1','confirmation_2'):
            actual=folder/(split+'_split.json');old=oldrun/(split+'_split.json')
            if read(actual)!=read(old):raise ValueError('Actual historical split changed: '+name+'/'+split)
            split_shas[split]=sha(actual)
        entries.append(dict(task=name,training=str(copied),training_sha256=sha(copied),
            features=str(features),features_sha256=sha(features),rows=len(task.y_train),
            task_spec=spec,capabilities=caps,registered_split_sha256=split_shas))
    save(output/'results.json',dict(entries=entries,source_files=frozen))
    return dict(status='prepared',actual_classifier_fits=0,real_API_requests=0,source_unchanged=hashes(ROOT)==frozen)


def execute(output,prepared,reuse_completed=None):
    registration=read(prepared/'results.json');frozen=hashes(ROOT)
    if frozen!=registration['source_files']:raise ValueError('Source differs from prepared inputs')
    results=[];env=environment(output);save(output/'checkpoint.json',results)
    prior_fits=physical(reuse_completed) if reuse_completed else []
    if reuse_completed and read(reuse_completed/'protocol.json')['source_files']!=frozen:
        raise ValueError('Previously completed source differs')
    for entry in registration['entries']:
        name=entry['task'];folder=output/name;folder.mkdir();run=folder/'run'
        old=reuse_completed/name if reuse_completed else None
        if old and (old/'run/run_outcome.json').is_file():
            oldrun=old/'run';outcome=read(oldrun/'run_outcome.json');manifest=read(oldrun/'run_manifest.json')
            selection=read(oldrun/'final_selection.json');fits=[r for r in prior_fits if r['run_name']==name]
            if (not outcome['requested_planner_completed'] or outcome['new_api_requests'] or
                manifest['task_archive_sha256']!=entry['training_sha256'] or manifest['capabilities']!=entry['capabilities'] or
                len(fits)!=outcome['classifier_fit_accounting']['attempted_classifier_fits'] or
                sha(oldrun/selection['model_file'])!=selection['model_sha256']):
                raise ValueError('Previous complete path identity differs')
            for split,digest in entry['registered_split_sha256'].items():
                if sha(oldrun/(split+'_split.json'))!=digest:raise ValueError('Previous split identity differs')
            expected=(old/'native.csv').read_bytes()
            if expected!=(old/'relocated.csv').read_bytes() or expected!=(old/'isolated.csv').read_bytes():
                raise ValueError('Previous independent predictions differ')
            with (old/'native.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            result=dict(task=name,reused_from=str(old),actual_classifier_fits=0,reused_classifier_fits=len(fits),
                outcome=outcome,selected_candidate=selection['family'],rows=len(rows),model_sha256=selection['model_sha256'],
                native_relocated_portable_identical=True,prediction_sha256=sha(old/'native.csv'),
                external_resources=read(old/'portable/bundle/bundle_manifest.json')['external_resources'])
            results.append(result);save(output/'checkpoint.json',results);print(dict(task=name,reused_completed_fits=len(fits)),flush=True)
            continue
        if sha(Path(entry['training']))!=entry['training_sha256'] or sha(Path(entry['features']))!=entry['features_sha256']:
            raise ValueError('Registered inputs changed')
        local={**env,'AB_A29_RUN_NAME':name}
        command=[sys.executable,'-m','superagent','run',entry['training'],'--run-dir',str(run),
            '--planner','local','--scheduler','ab-fixed','--portfolio','ab-v1','--catalog-scope','full',
            '--preference-catalog','standard','--parameter-semantics','off','--fit-limit','11','--seed','707',
            '--trial-timeout','180','--time-limit','900','--partial-delivery','checkpoint']
        save(folder/'command.json',command)
        training=child(command,folder/'train.log',local,960)
        outcome=read(run/'run_outcome.json');manifest=read(run/'run_manifest.json')
        fits=[r for r in physical(output) if r['run_name']==name]
        counts=outcome['classifier_fit_accounting']
        if (not outcome['requested_planner_completed'] or outcome['new_api_requests'] or
            len(fits)!=counts['attempted_classifier_fits'] or not 2<=len(fits)<=11 or
            counts['attempted_classifier_fits']!=counts['completed_classifier_fits']):
            raise ValueError('Path completion/counts mismatch: '+name)
        if manifest['capabilities']!=entry['capabilities']:raise ValueError('Runtime menu differs')
        for split,digest in entry['registered_split_sha256'].items():
            if sha(run/(split+'_split.json'))!=digest:raise ValueError('Runtime actual row split differs')
        selection=read(run/'final_selection.json')
        if sha(run/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model hash mismatch')
        features=entry['features'];native=folder/'native.csv'
        child([sys.executable,'-m','scripts.a38_predict_guard',features,'--run-dir',str(run),'--output',str(native)],folder/'native.log',local)
        moved=folder/'relocated';moved.mkdir()
        for file in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(run/file,moved/file)
        child([sys.executable,'-m','scripts.a38_predict_guard',features,'--run-dir',str(moved),'--output',str(folder/'relocated.csv')],folder/'relocated.log',local)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(folder/'portable')],folder/'bundle.log',local)
        bundle=folder/'portable/bundle'
        child([sys.executable,'-I',str(bundle/'predict.py'),'--features',features,'--output',str(folder/'isolated.csv')],folder/'isolated.log',local)
        expected=native.read_bytes()
        if expected!=(folder/'relocated.csv').read_bytes() or expected!=(folder/'isolated.csv').read_bytes():
            raise ValueError('Independent prediction bytes differ')
        with native.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        if [int(r['row_id']) for r in rows]!=list(range(len(rows))):raise ValueError('Prediction row order differs')
        result=dict(task=name,training=training,actual_classifier_fits=len(fits),outcome=outcome,
                    selected_candidate=selection['family'],rows=len(rows),model_sha256=selection['model_sha256'],
                    native_relocated_portable_identical=True,prediction_sha256=sha(native),
                    external_resources=read(bundle/'bundle_manifest.json')['external_resources'])
        results.append(result);save(output/'checkpoint.json',results)
        print(json.dumps({k:result[k] for k in ('task','actual_classifier_fits','selected_candidate','rows','native_relocated_portable_identical')}),flush=True)
    if hashes(ROOT)!=frozen or len(physical(output))+len(prior_fits)>33:raise ValueError('Source or aggregate budget changed')
    save(output/'results.json',results)
    return dict(status='completed',paths=3,actual_classifier_fits=len(physical(output)),real_API_requests=0,
        reused_classifier_fits=len(prior_fits),A_cumulative_fits_after=1505+len(physical(output))+len(prior_fits),source_unchanged=True,outer_labels_scored=False,
        evidence_scope='Current-source autonomous engineering acceptance on seen tasks, not unseen transfer or new C1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','execute'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepared',type=Path)
    p.add_argument('--reuse-completed',type=Path,help='Explicitly revalidate complete paths from a stopped wrapper; never retrain them');a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A122-'+a.phase,source_files=hashes(ROOT),runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),starting_A_fits=1505,
        maximum_classifier_fits=0 if a.phase=='prepare' else 33,maximum_real_API_requests=0,
        reuse_completed=str(a.reuse_completed) if a.reuse_completed else None,
        actual_row_partitions_reused=True,seen_development_tasks=True))
    try:
        result=prepare(a.output) if a.phase=='prepare' else execute(a.output,a.prepared,a.reuse_completed)
        save(a.output/'completion.json',dict(**result,elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        prior_count=len(physical(a.reuse_completed)) if a.reuse_completed else 0
        save(a.output/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,
            actual_classifier_fits=len(physical(a.output)),real_API_requests=0,
            reused_classifier_fits=prior_count,
            A_cumulative_fits_after=1505+prior_count+len(physical(a.output)),elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
