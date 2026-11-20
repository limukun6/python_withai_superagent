"""Offline, sequential, hard-bounded CLI delivery study with preserved source roots."""
import argparse
import csv
import hashlib
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
from a_batch_common import Journal,save,hashes

ROOT=Path(__file__).resolve().parents[1]


def pack(path,kind,x,y,metric,split):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        m={'schema_version':1,'objective':'classification','modality':{'single_text':'text','image':'image'}.get(kind,'tabular'),
           'metric':metric,'splits':{split:{'format':'csv' if kind=='single_text' else 'npz','path':split+('.csv' if kind=='single_text' else '.npz')}}}
        if kind=='single_text':
            m.update(columns=[{'name':'text','type':'text'}],target='label')
            b=io.StringIO(newline='');w=csv.writer(b);w.writerow(['text','label'] if y is not None else ['text'])
            for i,v in enumerate(x):w.writerow([v,y[i]] if y is not None else [v])
            z.writestr(split+'.csv',b.getvalue())
        else:
            b=io.BytesIO();np.savez_compressed(b,**({'X':x,'y':y} if y is not None else {'X':x}));z.writestr(split+'.npz',b.getvalue())
        z.writestr('manifest.json',json.dumps(m))


def prepare(a,j):
    sys.path.insert(0,str(ROOT))
    from superagent.data import load_training_task,TrainingTask
    from superagent.splits import grouped_partition,take,overlap_audit
    from superagent.models import search_candidates
    from sklearn.datasets import load_digits
    tasks=[]
    for name,file in [('bean','dry_bean_numeric_tabular.zip'),('sms','sms_spam_single_text.zip'),('digits',None)]:
        if file:
            path=Path(a.data_dir)/file;opened=[];original=zipfile.ZipFile.read
            from unittest.mock import patch
            def train_read(archive,member,*args,**kwargs):
                value=member.filename if isinstance(member,zipfile.ZipInfo) else member
                if any(t in value.lower() for t in ('validation','test','val.csv')):raise ValueError('Preparation attempted to read non-training data')
                opened.append(value);return original(archive,member,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',new=train_read):task=load_training_task(path)
            source={'package_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'opened_members':opened}
        else:
            data=load_digits();task=TrainingTask('image',data.images.astype(np.float32),data.target,{'metric':'accuracy'})
            source={'dataset':'sklearn.datasets.load_digits','shape':list(data.images.shape),'scope':'real low-resolution grayscale digits; development only',
                    'content_sha256':hashlib.sha256(task.X_train.tobytes()+task.y_train.tobytes()).hexdigest()}
        train,held,resource=grouped_partition(task,45101);folder=j.out/name;folder.mkdir()
        metric=task.spec.get('metric','accuracy')
        pack(folder/'train.zip',task.kind,take(task.X_train,train),task.y_train[train],metric,'train')
        pack(folder/'features.zip',task.kind,take(task.X_train,held),None,metric,'test')
        joblib.dump({'X':take(task.X_train,held),'y':task.y_train[held],'row_ids':held.tolist()},folder/'holdout.joblib')
        catalog=[{'id':c.family,'params':c.params} for c in search_candidates(task.kind,'expanded')]
        split={'train_rows':train.tolist(),'held_rows':held.tolist(),'resource':resource,'audit':overlap_audit(task,train,held)}
        save(folder/'partition.json',split)
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
        tasks.append({'name':name,'kind':task.kind,'metric':metric,'source':source,'catalog':catalog,'files':files,'train_rows':len(train),'held_rows':len(held)})
    save(j.out/'tasks.json',tasks);return tasks


def command_run(command,cwd,log,timeout):
    env={**os.environ,'PYTHONIOENCODING':'utf-8','SUPERAGENT_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    started=time.monotonic()
    with open(log,'w',encoding='utf-8') as f:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            child.wait(timeout=15);return {'status':'timeout','returncode':None,'seconds':time.monotonic()-started}
    return {'status':'completed' if code==0 else 'failed','returncode':code,'seconds':time.monotonic()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','run'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--data-dir');p.add_argument('--prepared');p.add_argument('--baseline-root');p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();j=Journal(a.output,ROOT,{'batch':'A-07','phase':a.phase,'baseline_root':a.baseline_root,'fit_limit':35,
        'per_path_fit_limit':7,'single_fit_timeout':60,'path_hard_timeout':300,'validate_only':a.validate_only})
    if a.validate_only:j.finish([],classifier_fits=0);return
    if a.phase=='prepare':j.finish(prepare(a,j),classifier_fits=0);return
    prepared=Path(a.prepared);tasks={r['name']:r for r in json.loads((prepared/'tasks.json').read_text(encoding='utf-8'))}
    roots={'baseline':Path(a.baseline_root).resolve(),'fixed':ROOT}
    frozen={k:hashes(v) for k,v in roots.items()};results=[];total=0
    for name,arm in [('bean','baseline'),('bean','fixed'),('sms','fixed'),('sms','baseline'),('digits','fixed')]:
        for key,root in roots.items():
            if hashes(root)!=frozen[key]:raise ValueError('Source changed during experiment')
        taskdir=prepared/name;meta=tasks[name]
        for file,digest in meta['files'].items():
            if hashlib.sha256((taskdir/file).read_bytes()).hexdigest()!=digest:raise ValueError('Prepared data changed')
        run=j.out/f'{name}_{arm}';source=roots[arm]
        command=[sys.executable,'-m','superagent','run',str(taskdir/'train.zip'),'--run-dir',str(run),
                 '--planner','local','--policy','interleaved_routes','--trials','2','--seed','707','--anchor','catalog',
                 '--portfolio','expanded','--trial-timeout','60','--time-limit','300']
        save(j.out/f'{name}_{arm}_launch.json',{'command':command,'source_files':frozen[arm],'task':meta})
        result={'task':name,'arm':arm,'source_root':str(source),**command_run(command,source,j.out/f'{name}_{arm}.log',300)}
        outcome=json.loads((run/'run_outcome.json').read_text(encoding='utf-8')) if (run/'run_outcome.json').exists() else None
        result['outcome']=outcome
        if outcome:
            accounting=outcome['classifier_fit_accounting'];result['classifier_fit_accounting']=accounting
            fits=accounting['completed_classifier_fits'];total+=fits
            if fits!=7 or not outcome['requested_planner_completed'] or outcome['new_api_requests']!=0:
                result['status']='failed'
        if result['status']!='completed':
            results.append(result);j.finish(results,completed_classifier_fits=total,attempted_classifier_fits=None,stopped_on_failure=True);return
        prediction_dir=run/'reloaded';prediction_dir.mkdir();csv_path=prediction_dir/'predictions.csv'
        pred=[sys.executable,'-m','superagent','predict',str(taskdir/'features.zip'),'--run-dir',str(run),'--output',str(csv_path)]
        reload_result=command_run(pred,source,prediction_dir/'predict.log',60)
        verify=[sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(source),'--run-dir',str(run),
                '--holdout',str(taskdir/'holdout.joblib'),'--predictions',str(csv_path),'--output',str(run/'export_verification.json')]
        verified=command_run(verify,source,prediction_dir/'verify.log',60) if reload_result['status']=='completed' else reload_result
        result['reload']=reload_result;result['verification_process']=verified
        if verified['status']=='completed':result['verification']=json.loads((run/'export_verification.json').read_text(encoding='utf-8'))
        else:result['status']='failed'
        results.append(result);save(j.out/'checkpoint.json',{'runs':results,'completed_classifier_fits':total})
        print(json.dumps({'task':name,'arm':arm,'status':result['status'],'fits':fits,'seconds':result['seconds'],'verification':result.get('verification')}),flush=True)
        if result['status']!='completed':break
    j.finish(results,completed_classifier_fits=total,attempted_classifier_fits=total,
             all_paths_passed=len(results)==5 and all(r['status']=='completed' for r in results))


if __name__=='__main__':main()
