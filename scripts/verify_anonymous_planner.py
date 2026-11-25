"""Freeze the source, run the default course entry without stdin, then score frozen outputs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split

from .unified_released_study import ROOT, archive, env, make_zip, read, require, save, sha, title_rows
from .verify_unified_raw import MEMBERS, feature_package
from superagent.ab_run import sources
from superagent.experiment_policy import identity


CASES = ('d01','d02','table','titles')


def revision():
    return identity(dict(core=sources(),entry=sha(ROOT/'unified_agent.py'),study=sha(Path(__file__))))


def array_bytes(x,y=None):
    out=io.BytesIO()
    np.savez(out,X=x,**({} if y is None else {'y':y}))
    return out.getvalue()


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    for case,dataset in (('d01',load_wine()),('d02',load_digits())):
        folder=a.out/case
        folder.mkdir()
        x=dataset.images if case=='d02' else dataset.data
        y=dataset.target
        fit,held=train_test_split(np.arange(len(y)),random_state=8291,test_size=.25,stratify=y)
        make_zip(folder/'task.zip',[('train.npz',array_bytes(x[fit],y[fit]))])
        make_zip(folder/'features.zip',[('test.npz',array_bytes(x[held]))])
        np.save(folder/'evaluation.npy',y[held],allow_pickle=False)
    for case in CASES[2:]:
        (a.out/case).mkdir()
    save(a.out/'protocol.json',dict(source=revision(),entry='python -m unified_agent train ARCHIVE --run-dir FRESH',
         budget=dict(fits_including_full_refit=10,model_api_requests=0),cases=list(CASES),
         development='Anonymous numeric 3-class and image 10-class public development probes; separate 25% panels.',
         packages={c:sha(package(a,c)) for c in CASES},baseline={'table':[1532,2030],'titles':[689,940]},
         evaluation='Freeze all models before reading external truth. Two independent prediction processes. Compare both released '
                    'scores once. Keep incumbent delivery if either score declines. No test members or feedback to the planner.'))


def package(a,c):
    return archive(a.course,c) if c in MEMBERS else a.out/c/'task.zip'


def execute(args, path, environment):
    with path.open('x') as stream:
        run=subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=ROOT,env=environment,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=1300)
    require(run.returncode==0,'Course entry failed. See '+str(path))


def run_case(a,c):
    protocol=read(a.out/'protocol.json')
    source=package(a,c)
    require(protocol['source']==revision() and sha(source)==protocol['packages'][c],'Frozen source or input differs')
    folder=a.out/c
    log=folder/'physical.jsonl'
    e=env(a.weights,log)
    member_file=folder/'opened.jsonl'
    expected=MEMBERS.get(c,['train.npz'])
    e.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))),
             UNIFIED_RAW_INPUT=str(source.resolve()),UNIFIED_RAW_LOG=str(member_file),
             UNIFIED_RAW_MEMBERS=json.dumps(expected))
    execute(['train',source,'--run-dir',folder/'run'],folder/'training.log',e)
    native=folder/'run/agent'
    sel,out=read(native/'final_selection.json'),read(native/'run_outcome.json')
    physical=[json.loads(x)['candidate'] for x in log.read_text().splitlines()]
    opened=set(json.loads(x) for x in member_file.read_text().splitlines())
    require(opened==set(expected) and out['new_api_requests']==0 and
            len(physical)==out['attempted_classifier_fits']<=10 and out['unknown_failed_attempts']==0,'Evidence mismatch')
    decisions=read(native/'decisions.json')
    require(decisions[-1]['choice']['action']=='stop' and sel['model_sha256']==sha(native/'selected_model.joblib')
            and protocol['source']==revision(),'Model, policy or final action mismatch')
    receipt=dict(model=sel['model_sha256'],members=out['members'],physical_fits=physical,
                 read_members=sorted(opened),api_calls=0,wall_seconds=out['wall_seconds'],
                 decisions=[{k:row['choice'][k] for k in ('action','members','split')} for row in decisions])
    save(folder/'sealed.json',receipt)
    print(c,json.dumps({k:receipt[k] for k in ('members','wall_seconds')},ensure_ascii=False),flush=True)


def score(a):
    require(read(a.out/'protocol.json')['source']==revision(),'Production code changed')
    records={c:read(a.out/c/'sealed.json') for c in CASES}
    frozen={}
    for c in CASES:
        folder=a.out/c
        require(records[c]['model']==sha(folder/'run/agent/selected_model.joblib'),'Model changed')
        if c in MEMBERS:
            feature_package(a.course,c,folder/'features.zip')
        prediction_digests=[]
        for repeat in range(2):
            dest=folder/('pred_'+str(repeat))/'predictions.csv'
            execute(['predict',folder/'features.zip','--run-dir',folder/'run','--output',dest],
                    folder/('prediction_'+str(repeat)+'.log'),env(a.weights,folder/'unexpected_fit.jsonl'))
            prediction_digests.append(sha(dest))
        require(len(set(prediction_digests))==1 and not (folder/'unexpected_fit.jsonl').exists(),'Reload changed predictions')
        frozen[c]=prediction_digests[0]
    save(a.out/'sealed-predictions.json',frozen)
    results={}
    for c in CASES:
        folder=a.out/c
        if c=='titles':
            truth=np.array([r[2] for r in title_rows(package(a,c),'val',labels=True)])
        elif c=='table':
            with zipfile.ZipFile(package(a,c)) as z:
                truth=np.load(io.BytesIO(z.read('in_vehicle_coupon_recommendation/y_val.npy')),allow_pickle=False)
        else:
            truth=np.load(folder/'evaluation.npy',allow_pickle=False)
        with (folder/'pred_0/predictions.csv').open() as file:
            rows=list(csv.DictReader(file))
        require([r['row_id'] for r in rows]==list(map(str,range(len(truth)))),'Misaligned prediction rows')
        hits=int(np.sum(np.array([r['prediction'] for r in rows],dtype=truth.dtype)==truth))
        results[c]=dict(correct=hits,rows=len(truth),accuracy=hits/len(truth),model=records[c]['model'],
                        prediction=frozen[c],fits=len(records[c]['physical_fits']),members=records[c]['members'])
    save(a.out/'results.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','score'])
    parser.add_argument('--case',choices=CASES)
    for name in ('out','course','weights'):
        parser.add_argument('--'+name,type=lambda s:Path(s).expanduser().resolve(),required=True)
    args=parser.parse_args()
    if args.stage=='train':
        if args.case is None:
            parser.error('train needs --case')
        run_case(args,args.case)
    elif args.stage=='prepare':
        prepare(args)
    else:
        score(args)
