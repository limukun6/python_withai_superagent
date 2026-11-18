"""Freeze both policies before training; open validation only after every model is frozen."""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE=Path(__file__).resolve()
def source_root():
    if (HERE.parents[1]/'superagent').is_dir():return HERE.parents[1]
    return HERE.parents[1]/'superagent_v42/source'
SRC=source_root()
sys.path.insert(0,str(SRC))
os.environ.update(SUPERAGENT_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
guard_dir=HERE.parent/'offline_guard'
os.environ['PYTHONPATH']=os.pathsep.join(map(str,[guard_dir,SRC]))
def guard(event,args):
    if event=='socket.connect' or event=='open' and isinstance(args[0],str) and '/api-test/' in args[0]:
        raise RuntimeError('Offline validation run')
sys.addaudithook(guard)
import joblib
import numpy as np
from superagent.adaptive_local import train,save,sha
from superagent.benchmark import run_one
from superagent.data import load_training_task,DevelopmentTask
from superagent.models import predict_with_model

TASKS={'task1':'Released_Task1-In_Vehicle_Coupon_Recommendation.zip','task2':'Released_Task2-Headlines_CTR.zip'}
ARMS=('fixed','adaptive')
def read(path):return json.loads(path.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def hashes():
    return {'core':{p.relative_to(SRC).as_posix():sha(p) for p in sorted((SRC/'superagent').glob('*.py'))},
        'runner':sha(HERE),'worker_network_guard':sha(guard_dir/'sitecustomize.py')}
@contextmanager
def allow_only(names,log,phase):
    orig=zipfile.ZipFile.open
    def checked(archive,member,*args,**kwargs):
        name=member.filename if isinstance(member,zipfile.ZipInfo) else member
        if Path(name).name not in names:raise ValueError('Unapproved released member: '+name)
        with log.open('a') as stream:print(json.dumps(dict(phase=phase,member=name,utc=now())),file=stream)
        return orig(archive,member,*args,**kwargs)
    zipfile.ZipFile.open=checked
    try:yield
    finally:zipfile.ZipFile.open=orig
def run_task(packages,root,name):
    directory=root/name;directory.mkdir()
    archive=packages/TASKS[name]
    with allow_only({'C_train.npy','y_train.npy','train.csv'},directory/'access.jsonl','training'):
        task=load_training_task(archive)
    results={}
    for arm in ARMS:
        path=directory/arm;start=time.monotonic()
        if arm=='fixed':
            tmp=DevelopmentTask(task.kind,'released_training_only',task.X_train,task.y_train,None,None,task.spec)
            result=run_one(tmp,'local_interleaved_routes',707,path,None,max_trials=4,time_limit=300,trial_timeout=90,
                anchor_first=True,refit_full_train=True,confirmation_repeats=3,confirmation_top_k=3,
                confirmation_grouping='shared_text',search_grouping='shared_text',portfolio='expanded',feature_cache=True)
            assert result['valid_policy_run'] and not result['failed_trials'] and result['trials_attempted']==4
            assert result['confirmation']['valid_confirmation'] and result['refit_succeeded'] and result['llm_calls']==0
            fits=result['classifier_fit_accounting']['completed_classifier_fits'];selected=result['selected_candidate']
        else:
            result=train(task,path);fits=result['fits'];selected=result['selected']
        selection=read(path/'final_selection.json')
        assert selection['refit'] and not selection['test_accessed'] and selection['family']==selected
        assert selection['training_rows']==len(task.y_train) and selection['model_sha256']==sha(path/'selected_model.joblib')
        results[arm]=dict(selected=selected,fits=fits,elapsed_seconds=time.monotonic()-start,
             train_rows=len(task.y_train),model_sha256=selection['model_sha256'],frozen_utc=now(),model_api_calls=0)
        print(name,arm,selected,fits,flush=True)
    save(directory/'frozen.json',results)
def validation(archive):
    with zipfile.ZipFile(archive) as zipped:
        def one(name):
            member,=[s for s in zipped.namelist() if not s.startswith('__MACOSX/') and Path(s).name==name]
            return zipped.read(member)
        if any(Path(s).name=='C_val.npy' for s in zipped.namelist()):
            return np.load(io.BytesIO(one('C_val.npy')),allow_pickle=True),np.load(io.BytesIO(one('y_val.npy')),allow_pickle=True)
        rows=list(csv.DictReader(io.StringIO(one('val.csv').decode('utf-8-sig'))))
        return [(r['headline_A'],r['headline_B']) for r in rows],np.array([int(r['label_pairwise']) for r in rows])
def score(packages,root):
    frozen={n:read(root/n/'frozen.json') for n in TASKS}
    for name,arms in frozen.items():
        assert set(arms)==set(ARMS)
        for arm,row in arms.items():assert sha(root/name/arm/'selected_model.joblib')==row['model_sha256']
    (root/'scoring_started').touch(exist_ok=False)
    results={}
    for name in TASKS:
        directory=root/name
        with allow_only({'C_val.npy','y_val.npy','val.csv'},directory/'access.jsonl','validation_after_four_freezes'):
            features,truth=validation(packages/TASKS[name])
        masks={};preds={};arms={}
        for arm in ARMS:
            dest=directory/arm;bundle=joblib.load(dest/'selected_model.joblib')
            guess=np.asarray(predict_with_model(bundle['kind'],bundle['model'],features)).reshape(-1)
            assert guess.shape==truth.shape
            masks[arm]=guess==truth;preds[arm]=guess
            with (dest/'validation_predictions.csv').open('w',newline='') as file:
                writer=csv.writer(file);writer.writerow(['row_id','prediction']);writer.writerows(enumerate(guess.tolist()))
            arms[arm]=dict(**frozen[name][arm],correct=int(masks[arm].sum()),accuracy=float(masks[arm].mean()),
                           prediction_sha256=sha(dest/'validation_predictions.csv'))
        fixed,adaptive=masks.values()
        results[name]=dict(rows=len(truth),arms=arms,fixed_by_adaptive=int((~fixed&adaptive).sum()),
            lost_by_adaptive=int((fixed&~adaptive).sum()),different_predictions=int((preds['fixed']!=preds['adaptive']).sum()))
    save(root/'scores.json',results);print(json.dumps(results),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze',*TASKS,'score'])
    parser.add_argument('--packages',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve();packages=args.packages.resolve()
    if args.phase=='freeze':
        root.mkdir(parents=True,exist_ok=False)
        save(root/'protocol.json',dict(hashes=hashes(),frozen_utc=now(),seed=707,cap_fits=11,seconds=300,
            primary_submission_arm='fixed',challenger='adaptive',catalog='expanded',search='interleaved_routes',
            selection='Never use validation to choose between arms. Score both after four freezes. One run per arm.',
            metric='accuracy',split='provided development validation',api_calls_allowed=0))
    else:
        assert hashes()==read(root/'protocol.json')['hashes']
        if args.phase=='score':score(packages,root)
        else:run_task(packages,root,args.phase)
