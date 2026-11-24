"""Bounded merged-core official runs and frozen source-bundle delivery, no API/H."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import joblib
import numpy as np
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import launch, physical
from scripts.run_b1_ab_acceptance import packages
from superagent.ab_run import sources
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model, predict_with_model
from superagent.splits import take
from tools.export_model_bundle import export

ROOT=Path(__file__).resolve().parents[1]


def csv_package(path, rows, semantics, split='train', labels=True):
    stream=io.StringIO(newline=''); writer=csv.writer(stream)
    writer.writerow(['first','second']+(['label'] if labels else [])); writer.writerows(rows)
    manifest=dict(schema_version=1,objective='classification',modality='pair_text',metric='accuracy',
                  columns=[dict(name=n,type='text') for n in ('first','second')],target='label',
                  pair_semantics=semantics,splits={split:dict(format='csv',path='rows.csv')})
    if semantics=='symmetric': manifest['pair_label_one']='equivalent'
    with zipfile.ZipFile(path,'x') as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('rows.csv',stream.getvalue())


def prepare(root, reference):
    folder=root/'numeric';folder.mkdir();packages(folder)
    tasks=[('numeric',folder/'train.zip',folder/'test.zip')]
    for semantics in ('symmetric','relation'):
        folder=root/semantics;folder.mkdir();rows=[]
        for i in range(180):
            label=i%(2 if semantics=='symmetric' else 3)
            left=f'item{i} topic{i%7} bright apple';right=f'peer{i} topic{i%7} '+('bright apple' if label else 'dark river')
            rows.append([left,right,str(label)])
        csv_package(folder/'train.zip',rows,semantics)
        csv_package(folder/'test.zip',[r[:2] for r in rows[:31]],semantics,'test',False)
        tasks.append((semantics,folder/'train.zip',folder/'test.zip'))
    entry=read(reference/'protocol.json')['entries']['sms'];src=Path(entry['training_source'])
    if sha(src)!=entry['training_source_sha256']:raise ValueError('SMS training source changed')
    folder=root/'sms_auc';folder.mkdir()
    with zipfile.ZipFile(src) as original,zipfile.ZipFile(folder/'train.zip','x') as dest:
        if set(original.namelist())!={'train.csv','manifest.json'}:raise ValueError('Expected training-only SMS input')
        for name in original.namelist():
            content=original.read(name)
            if name=='manifest.json':
                manifest=json.loads(content)
                if set(manifest['splits'])!={'train'}:raise ValueError('Only training input allowed')
                manifest.update(metric='roc_auc',positive_label='spam');content=json.dumps(manifest).encode()
            dest.writestr(name,content)
    with zipfile.ZipFile(folder/'train.zip') as original,zipfile.ZipFile(folder/'test.zip','x') as dest:
        manifest=json.loads(original.read('manifest.json'));desc=manifest['splits']['train']
        text=original.read(desc['path']).decode('utf-8-sig');reader=csv.reader(io.StringIO(text));header=next(reader)
        target=header.index(manifest['target']);indices=[i for i in range(len(header)) if i!=target]
        data=io.StringIO(newline='');writer=csv.writer(data);writer.writerow([header[i] for i in indices])
        writer.writerows([[row[i] for i in indices] for _,row in zip(range(31),reader)])
        manifest['splits']={'test':dict(format='csv',path='rows.csv')}
        dest.writestr('manifest.json',json.dumps(manifest));dest.writestr('rows.csv',data.getvalue())
    tasks.append(('sms_auc',folder/'train.zip',folder/'test.zip'))
    return tasks


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    env={**os.environ,'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
         'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'AB_A29_RUN_NAME':'unassigned',
         'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),
         'PYTHONIOENCODING':'utf-8','SUPERAGENT_OFFLINE':'1','SUPERAGENT_MINILM_WEIGHTS':'',
         'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    save(root/'protocol.json',dict(batch='A74',base_commit='9c7ceaf',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A74_merged_delivery_plan.md'),maximum_classifier_fits=14,starting_A_fits=1214,
        real_api_requests=0,formal_test_or_H_opened=False,performance_superiority_test=False))
    try:
        tasks=prepare(root,a.reference)
        for name,train,query in tasks:
            directory=root/name/'run';used=len(physical(root));active={**env,'AB_A29_RUN_NAME':name}
            code,seconds=launch([sys.executable,'-m','superagent','run',str(train),'--run-dir',str(directory),
                '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','3','--seed','707',
                '--time-limit','600','--trial-timeout','180','--partial-delivery','checkpoint'],root/name/'train.log',active,660)
            outcome=read(directory/'run_outcome.json')
            if code or not outcome['requested_planner_completed']:raise RuntimeError(name+' official run failed: '+str(outcome.get('error')))
            count=len(physical(root))-used
            if count!=3 or outcome['classifier_fit_accounting']['completed_classifier_fits']!=3:raise ValueError('Fit counts differ')
            if read(directory/'run_manifest.json')['source_files']!=frozen:raise ValueError('Source differs')
            selected=read(directory/'final_selection.json')
            if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Model changed')
            prediction=root/name/'prediction.csv'
            code,_=launch([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(directory),
                '--output',str(prediction)],root/name/'predict.log',active,180)
            if code:raise RuntimeError(name+' new-process prediction failed')
            bundle=export(directory,root/name/'export');fresh=root/name/'new_cwd';fresh.mkdir()
            with (root/name/'isolated.log').open('w',encoding='utf-8') as log:
                code=subprocess.run([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(query),
                    '--output',str(root/name/'isolated.csv')],cwd=fresh,env=active,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            if code or prediction.read_bytes()!=(root/name/'isolated.csv').read_bytes():raise ValueError(name+' isolated prediction differs')
            if len(physical(root))-used!=3:raise ValueError('Inference trained a classifier')
            row=dict(task=name,actual_classifier_fits=count,process_seconds=seconds,selected_candidate=selected['family'],
                probability_csv_byte_identical=True,official_new_process_prediction=True,source_bundle_prediction=True,
                primary_metric=outcome['primary_metric'],prediction_rows=31,real_api_requests=outcome['new_api_requests'])
            results.append(row);save(root/'results.json',results)
        for name,candidate in [('symmetric','unordered_word_char'),('relation','ordered_text_pair_interactions')]:
            task=load_training_task(root/name/'train.zip');folder=root/(name+'_B_action');folder.mkdir();payload=folder/'payload.joblib'
            joblib.dump(dict(kind=task.kind,pair_semantics=name,X_train=task.X_train,y_train=task.y_train,
                X_search=take(task.X_train,np.arange(31)),y_search=task.y_train[:31],metric_spec=MetricSpec.for_task(task).record()),payload)
            save(folder/'admission.json',dict(maximum_classifier_fits=1,ledger_path=str(folder/'fits.json')))
            used=len(physical(root));code,seconds=launch([sys.executable,'-m','superagent.experiment_worker',str(payload),candidate,
                str(folder/'model.joblib'),'--fit-admission',str(folder/'admission.json')],folder/'train.log',
                {**env,'AB_A29_RUN_NAME':name+'_B_action'},180)
            if code:raise RuntimeError('New B action failed: '+candidate)
            if len(physical(root))-used!=1 or read(folder/'fits.json')['completed_classifier_fits']!=1:raise ValueError('B action fit count differs')
            bundle=joblib.load(folder/'model.joblib');classes,prob=probability_with_model(task.kind,bundle['model'],task.X_train[:31])
            swap_error=None
            if name=='symmetric':
                _,reverse=probability_with_model(task.kind,bundle['model'],[(b,a) for a,b in task.X_train[:31]])
                swap_error=float(np.max(np.abs(prob-reverse)))
                if swap_error>1e-12:raise ValueError('B unordered action violates symmetry')
            elif len(classes)!=3:raise ValueError('Ordered relation lost a class')
            results.append(dict(task=name+'_B_action',candidate=candidate,actual_classifier_fits=1,process_seconds=seconds,
                                swap_probability_max_error=swap_error,class_count=len(classes),not_autonomous_performance_test=True))
            save(root/'results.json',results)
        if sources()!=frozen:raise ValueError('Source changed')
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=len(physical(root)),real_api_requests=0,
            A_cumulative_fits_after=1214+len(physical(root)),elapsed_seconds=time.monotonic()-start,
            source_unchanged=True,formal_test_or_H_opened=False,performance_superiority_test=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1214+len(physical(root)),real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
