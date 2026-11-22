"""Actual pinned frozen encoder probes plus one autonomous delivery path; bounded fits, offline."""
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
from sklearn.model_selection import StratifiedShuffleSplit
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from superagent.ab_run import sources
from superagent.ab_sentence import encode,verified_directory
from superagent.ab_vision import weights_file,FrozenVisionFeatures
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from tools.export_model_bundle import export

ROOT=Path(__file__).resolve().parents[1]


def package(path,texts,labels=None):
    split='train' if labels is not None else 'test';stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['message']+(['label'] if labels is not None else []))
    for i,text in enumerate(texts):writer.writerow([text]+([str(labels[i])] if labels is not None else []))
    manifest=dict(schema_version=1,objective='classification',modality='text',metric='macro_f1',
        columns=[dict(name='message',type='text')],target='label',splits={split:dict(format='csv',path='rows.csv')})
    with zipfile.ZipFile(path,'x') as z:z.writestr('manifest.json',json.dumps(manifest));z.writestr('rows.csv',stream.getvalue())


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    save(root/'protocol.json',dict(batch='A81',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A81_real_encoder_execution_plan.md'),source_files=frozen,
        environment=read(a.assets/'environment.json'),assets=read(a.assets/'assets.json'),maximum_classifier_fits=9,
        starting_A_fits=1239,real_api_requests=0,formal_test_or_H_opened=False,not_performance_superiority_test=True))
    env={**os.environ,'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
        'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'AB_A29_RUN_NAME':'unassigned',
        'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),'PYTHONIOENCODING':'utf-8',
        'SUPERAGENT_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A81 offline')
    sys.addaudithook(offline)
    try:
        if read(a.assets/'completion.json')['status']!='ready' or verified_directory() is None or weights_file() is None:
            raise ValueError('Pinned assets or dependencies unavailable')
        dense,_=encode(['known fixed encoder probe','another fixed encoder probe'])
        if dense.shape!=(2,384) or not np.isfinite(dense).all():raise ValueError('Real MiniLM features invalid')
        rng=np.random.default_rng(40811);images=rng.integers(0,256,(32,32,32,3),dtype=np.uint8)
        vision=FrozenVisionFeatures().fit(images);features=vision.transform(images[:2])
        if features.shape!=(2,576) or not np.isfinite(features).all():raise ValueError('Real vision features invalid')
        save(root/'smoke.json',dict(real_MiniLM_shape=list(dense.shape),real_vision_shape=list(features.shape),
            MiniLM_norms=np.linalg.norm(dense,axis=1).tolist(),vision_norms=np.linalg.norm(features,axis=1).tolist(),classifier_fits=0))
        entry=read(a.reference/'protocol.json')['entries']['sms'];src=Path(entry['training_source'])
        if sha(src)!=entry['training_source_sha256']:raise ValueError('SMS source changed')
        task=load_training_task(src);indices,_=next(StratifiedShuffleSplit(n_splits=1,train_size=256,random_state=40811).split(np.zeros(len(task.y_train)),task.y_train))
        texts=[task.X_train[i] for i in indices];labels=task.y_train[indices]
        package(root/'anonymous_train.zip',texts,labels);package(root/'anonymous_query.zip',texts[:31])
        load_training_task(root/'anonymous_train.zip')
        save(root/'sample_identity.json',dict(seed=40811,rows=len(indices),training_source_sha256=sha(src),row_indices=indices.tolist()))
        pair_labels=np.arange(64)%2;pairs=[]
        for i,label in enumerate(pair_labels):
            left,right=f'good apple item{i}',f'bad river peer{i}'
            pairs.append((left,right) if label else (right,left))
        cases=[('plain','single_text','minilm_text_with_lexical_features',texts,labels),
               ('nonlinear','single_text','minilm_text_with_nonlinear_lexical_features',texts,labels),
               ('preference','pair_text','minilm_text_with_nonlinear_lexical_features',pairs,pair_labels),
               ('vision','image','image_mobilenet_v3_small_linear',images,np.arange(32)%2)]
        for name,kind,candidate,rows,y in cases:
            folder=root/name;folder.mkdir();query=rows[:8];payload=dict(kind=kind,X_train=rows,y_train=y,X_search=query,y_search=y[:8],
                metric_spec=MetricSpec('accuracy',tuple(np.unique(y).tolist())).record())
            if kind=='pair_text':payload['pair_semantics']='preference'
            joblib.dump(payload,folder/'payload.joblib');save(folder/'admission.json',dict(maximum_classifier_fits=1,ledger_path=str(folder/'fits.json')))
            before=len(physical(root));code,seconds=launch([sys.executable,'-m','superagent.experiment_worker',str(folder/'payload.joblib'),candidate,
                str(folder/'model.joblib'),'--fit-admission',str(folder/'admission.json')],folder/'train.log',{**env,'AB_A29_RUN_NAME':name},180)
            if code:raise RuntimeError('Actual encoder worker failed: '+name)
            if len(physical(root))-before!=1:raise ValueError('Actual encoder fit count differs')
            fitted=joblib.load(folder/'model.joblib');classes,probs=probability_with_model(kind,fitted['model'],query);swap_error=None
            if kind=='pair_text':
                _,reverse=probability_with_model(kind,fitted['model'],[(b,a) for a,b in query]);positive=list(classes).index(1)
                swap_error=float(np.max(np.abs(probs[:,positive]+reverse[:,positive]-1)))
                if swap_error>1e-5:raise ValueError('Real encoder preference violates swap tolerance')
            results.append(dict(probe=name,candidate=candidate,actual_classifier_fits=1,process_seconds=seconds,
                swap_probability_error=swap_error,fitted_estimator=fitted['metrics']['estimator'],actual_assets_used=True))
            save(root/'results.json',results)
        directory=root/'run';before=len(physical(root))
        code,seconds=launch([sys.executable,'-m','superagent','run',str(root/'anonymous_train.zip'),'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','5','--seed','707','--time-limit','600',
            '--trial-timeout','180','--partial-delivery','checkpoint'],root/'train.log',{**env,'AB_A29_RUN_NAME':'official'},660)
        outcome=read(directory/'run_outcome.json')
        if code or not outcome['requested_planner_completed']:raise RuntimeError('Actual encoder autonomous entry failed: '+str(outcome.get('error')))
        menu=read(directory/'run_manifest.json')['capabilities']
        if not all(name in menu for name in ('minilm_text_with_lexical_features','minilm_text_with_nonlinear_lexical_features')):raise ValueError('Executable shared menu hides verified encoders')
        official_fits=len(physical(root))-before
        if official_fits>5 or official_fits!=outcome['classifier_fit_accounting']['attempted_classifier_fits']:raise ValueError('Official encoder fit counts differ')
        prediction=root/'prediction.csv';code,_=launch([sys.executable,'-m','scripts.a38_predict_guard',str(root/'anonymous_query.zip'),
            '--run-dir',str(directory),'--output',str(prediction)],root/'predict.log',env,180)
        if code:raise RuntimeError('Actual encoder new-process delivery failed')
        bundle=export(directory,root/'export');fresh=root/'new_cwd';fresh.mkdir()
        def isolated(label,environment):
            with (root/(label+'.log')).open('w',encoding='utf-8') as log:
                return subprocess.run([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(root/'anonymous_query.zip'),
                    '--output',str(root/(label+'.csv'))],cwd=fresh,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        if isolated('isolated',env) or (root/'isolated.csv').read_bytes()!=prediction.read_bytes():raise ValueError('Actual encoder source-bundle output differs')
        manifest=read(bundle/'bundle_manifest.json');missing_refused=None
        if manifest['external_resources']:
            missing={**env,'SUPERAGENT_MINILM_WEIGHTS':'','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':''}
            missing_refused=isolated('missing_weights',missing)!=0 and not (root/'missing_weights.csv').exists()
            if not missing_refused:raise ValueError('Actual encoder export allowed missing weights')
        if len(physical(root))!=before+official_fits:raise ValueError('Delivery fitted a classifier')
        results.append(dict(probe='official',actual_classifier_fits=official_fits,process_seconds=seconds,
            selected_candidate=read(directory/'final_selection.json')['family'],catalogue=list(menu),
            executed_candidates=[r['candidate'] for r in read(directory/'history.json')],CSV_byte_identical=True,
            external_resources=manifest['external_resources'],encoder_dependencies=manifest['dependencies'],missing_weights_refused=missing_refused))
        save(root/'results.json',results)
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=len(physical(root)),real_api_requests=0,
            A_cumulative_fits_after=1239+len(physical(root)),source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,
            formal_test_or_H_opened=False,not_performance_superiority_test=True))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),real_api_requests=0,
            A_cumulative_fits_after=1239+len(physical(root)),automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
