"""Run from this file: freeze, replay, task1, task2, report. Historical core remains unchanged."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'source';OUT=ROOT/'results'
OLD=ROOT.parent/'superagent_v41/results'
GUARD=ROOT.parent/'superagent_v38/offline_guard'
os.environ.update(PYTHONPATH=os.pathsep.join(map(str,(GUARD,SOURCE))),SUPERAGENT_OFFLINE='1',
    SUPERAGENT_FORBID_TEST_MEMBERS='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
    VECLIB_MAXIMUM_THREADS='2')
sys.path.insert(0,str(SOURCE))
def offline(event,args):
    if event=='socket.connect' or (event=='open' and isinstance(args[0],str) and '/api-test/' in args[0]):
        raise RuntimeError('Offline experiment; credentials disabled')
sys.addaudithook(offline)
import joblib
import numpy as np
from superagent.adaptive_local import train,save,sha
from superagent.evidence_schedule import catalog_for,schedule,duel
from superagent.local_search import LocalSearchPolicy
from superagent.evidence_agent import route_of
from superagent.data import load_training_task,TrainingTask,DevelopmentTask
from superagent.benchmark import run_one
from superagent.models import predict_with_model
from superagent.splits import grouped_partition,overlap_audit,take

SEEDS=(42011,42017,42029)
PACKAGES={'task1':'Released_Task1-In_Vehicle_Coupon_Recommendation.zip','task2':'Released_Task2-Headlines_CTR.zip'}
def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def manifest():
    files=[p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    return {os.path.relpath(p,ROOT):sha(p) for p in [*files,Path(__file__),ROOT/'PROTOCOL.txt',GUARD/'sitecustomize.py']}

def fixed_sim(catalog,measure):
    policy=LocalSearchPolicy('interleaved_routes',catalog,707);history=[]
    for _ in range(4):
        name,_=policy.choose(history)
        history.append(dict(candidate=name,route=route_of(name),status='completed',search_accuracy=float(measure(name,0).mean())))
    top=sorted(history,key=lambda row:(-row['search_accuracy'],row['candidate']))[:3]
    weighted={r['candidate']:np.mean([r['search_accuracy'],measure(r['candidate'],1).mean(),measure(r['candidate'],2).mean()]) for r in top}
    return min(weighted,key=lambda name:(-weighted[name],name))

class NoPastObservation(Exception):pass
def replay():
    output=[]
    for name in PACKAGES:
        for seed in (41113,41117,41131):
            folder=OLD/name/f'agent_{seed}'/'catalog'
            trials=read(folder/'trials.json');kind=read(folder/'summary.json')['task_kind']
            def measure(candidate,split):
                trial=next((r['trial'] for r in trials if r['candidate']==candidate),None)
                path=folder/(f'trial_{trial}.joblib' if split==0 else f'confirmation_{split}_{trial}.joblib')
                if not path.exists():raise NoPastObservation((candidate,split))
                payload=folder/('inner_data.joblib' if split==0 else f'confirmation_data_{split}.joblib')
                return joblib.load(path)['search_predictions']==joblib.load(payload)['y_search']
            fixed=fixed_sim(catalog_for(kind),measure)
            assert fixed==read(folder/'final_selection.json')['family']
            trace=[];missing=None;adaptive=None
            try:adaptive=schedule(catalog_for(kind),measure,trace.append)
            except NoPastObservation as exc:missing=list(exc.args[0])
            output.append(dict(task=name,seed=seed,fixed=fixed,reproduced=True,adaptive=adaptive,missing=missing,trace=trace))
    save(OUT/'replay.json',output)
    print('control matched',len(output),'adaptive fully observable',sum(r['missing'] is None for r in output),flush=True)

def load(name):
    original=zipfile.ZipFile.open;access=[]
    def guarded(archive,member,*args,**kwargs):
        filename=member.filename if isinstance(member,zipfile.ZipInfo) else member
        if Path(filename).name not in ('train.csv','C_train.npy','y_train.npy'):raise ValueError('Not a training member')
        access.append(filename)
        return original(archive,member,*args,**kwargs)
    zipfile.ZipFile.open=guarded
    try: task=load_training_task(Path.home()/'Downloads/Final_Project_export'/PACKAGES[name])
    finally:zipfile.ZipFile.open=original
    return task,access
def subset(task,indices):
    return TrainingTask(task.kind,take(task.X_train,indices),task.y_train[indices],task.spec)
def correct_audit(task,train_ids,held):
    result=overlap_audit(task,train_ids,held)
    assert not result.get('search_duplicate_rows',0) and not result.get('search_rows_sharing_text',0)
    return result
def run(name):
    root=OUT/name;root.mkdir(exist_ok=False)
    whole,access=load(name);save(root/'package_access.json',access)
    keep,reserve,_=grouped_partition(whole,41101,'shared_text')
    assert keep.tolist()==read(OLD/name/'reserve.json')['train'] and reserve.tolist()==read(OLD/name/'reserve.json')['held']
    dev=subset(whole,keep);rows=[]
    for index,seed in enumerate(SEEDS):
        fit,held,_=grouped_partition(dev,seed,'shared_text')
        folder=root/str(seed);folder.mkdir()
        save(folder/'outer.json',dict(train=keep[fit].tolist(),held=keep[held].tolist(),audit=correct_audit(dev,fit,held)))
        training=subset(dev,fit)
        arms={}
        for arm in (('fixed','adaptive') if index%2==0 else ('adaptive','fixed')):
            dest=folder/arm
            start=time.monotonic()
            try:
                if arm=='adaptive':
                    result=train(training,dest)
                    summary=dict(selected=result['selected'],fits=result['fits'])
                else:
                    task=DevelopmentTask(training.kind,'train_only',training.X_train,training.y_train,None,None,training.spec)
                    result=run_one(task,'local_interleaved_routes',707,dest,None,max_trials=4,time_limit=300,trial_timeout=90,
                        anchor_first=True,refit_full_train=True,confirmation_repeats=3,confirmation_top_k=3,
                        confirmation_grouping='shared_text',search_grouping='shared_text',portfolio='expanded',feature_cache=True)
                    if not (result['valid_policy_run'] and result['refit_succeeded'] and result['confirmation']['valid_confirmation']
                        and not result['failed_trials'] and result['trials_attempted']==4 and result['llm_calls']==0):raise RuntimeError('Incomplete control')
                    summary=dict(selected=result['selected_candidate'],fits=result['classifier_fit_accounting']['completed_classifier_fits'])
                summary.update(complete=True,elapsed=time.monotonic()-start,model_hash=sha(dest/'selected_model.joblib'))
            except Exception as exc:
                summary=dict(complete=False,error=str(exc),elapsed=time.monotonic()-start)
            arms[arm]=summary;save(folder/'frozen_arms.json',arms)
            print(name,seed,arm,summary,flush=True)
        predictions={}
        for arm,summary in arms.items():
            if not summary['complete']:continue
            model=folder/arm/'selected_model.joblib';assert sha(model)==summary['model_hash']
            bundle=joblib.load(model);assert bundle['candidate']==summary['selected']
            pred=predict_with_model(dev.kind,bundle['model'],take(dev.X_train,held))
            predictions[arm]=pred==dev.y_train[held]
            save(folder/f'{arm}_outer_predictions.json',np.asarray(pred).tolist())
            summary.update(correct=int(predictions[arm].sum()),accuracy=float(predictions[arm].mean()))
        row=dict(seed=seed,arms=arms,rows=len(held))
        if len(predictions)==2:row.update(duel(predictions['fixed'],predictions['adaptive']))
        rows.append(row);save(root/'paired.json',rows)
        assert manifest()==read(OUT/'freeze.json')
    save(root/'done.json',dict(pairs=len(rows)))
if __name__=='__main__':
    action=sys.argv[1]
    if action=='freeze':
        OUT.mkdir(exist_ok=False);save(OUT/'freeze.json',manifest())
    else:
        assert read(OUT/'freeze.json')==manifest()
        if action=='replay':replay()
        else:run(action)
