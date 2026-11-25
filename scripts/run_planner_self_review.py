"""Paired prospective experiment for the anonymous planner; evaluation reads only frozen outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from scripts.unified_released_study import ROOT, env, make_zip, read, require, save, sha
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec


PROTOCOL = ROOT/'research/rsi_planner_1125/protocol.json'
CASES = ('n17','n29','n43','n61')
ARMS = ('original','review')
OUTER_SEED = 602911
FIT_LIMIT = 9


def source_identity(root):
    files = [root/'unified_agent.py', *sorted((root/'superagent').rglob('*.py'))]
    return identity({p.relative_to(root).as_posix():sha(p) for p in files})


def datasets(sms):
    numeric = load_breast_cancer()
    yield CASES[0], 'tabular', 'balanced_accuracy', numeric.data, numeric.target
    with zipfile.ZipFile(sms) as z:
        records=[line.split('\t',1) for line in z.read('SMSSpamCollection').decode('utf-8-sig').splitlines() if line]
    yield CASES[1], 'text', 'macro_f1', np.asarray([r[1] for r in records]), np.asarray([r[0] for r in records])
    images=load_digits()
    yield CASES[2], 'image', 'accuracy', images.images, images.target
    rng=np.random.default_rng(134037)
    values=rng.integers(0,6,(1280,8))
    labels=(values[:,0]//2 + (values[:,1]==values[:,2])*2 + (values[:,3]>=3)*(values[:,4]//2))%3
    noisy=rng.random(len(labels))<.12
    labels[noisy]=rng.integers(0,3,size=noisy.sum())
    yield CASES[3], 'categorical', 'macro_f1', np.char.add('v',values.astype(str)), labels


def distinct(x,y):
    representatives,conflicts={},set()
    for i,row in enumerate(x):
        key=' '.join(str(row).casefold().split()) if x.ndim==1 else row.tobytes()
        if key in representatives and y[i]!=y[representatives[key]]:
            conflicts.add(key)
        else:
            representatives.setdefault(key,i)
    return np.asarray([i for key,i in representatives.items() if key not in conflicts])


def packet(path,modality,metric,x,y=None):
    role='train' if y is not None else 'test'
    raw=io.BytesIO()
    if modality in ('image','tabular'):
        np.savez_compressed(raw,X=x,**({'y':y} if y is not None else {}))
        fmt,content='npz',raw.getvalue()
        spec=dict(modality=modality)
    else:
        text=io.StringIO(newline='')
        writer=csv.writer(text)
        cols=['document'] if modality=='text' else [f'c{i}' for i in range(x.shape[1])]
        writer.writerow(cols+(['target'] if y is not None else []))
        for i,row in enumerate(x):
            writer.writerow(([row] if modality=='text' else list(row))+([y[i]] if y is not None else []))
        fmt,content='csv',text.getvalue().encode()
        spec=dict(modality='text' if modality=='text' else 'tabular',target='target',
                  columns=[dict(name=c,type='text' if modality=='text' else 'categorical') for c in cols])
    member=f'rows.{fmt}'
    manifest=dict(schema_version=1,metric=metric,splits={role:dict(format=fmt,path=member)},**spec)
    make_zip(path,[('manifest.json',json.dumps(manifest).encode()),(member,content)])


def prepare(a):
    require(read(a.root/'floor'/'autonomy.json')['source']==read(PROTOCOL)['submission_floor'],
            'Submission-floor audit must finish before this experiment')
    out=a.root/'paired'
    out.mkdir()
    sealed=out/'evaluation'
    sealed.mkdir()
    cases={}
    for case,modality,metric,x,y in datasets(a.sms):
        folder=out/case
        folder.mkdir()
        ids=distinct(x,y)
        if case=='n29':
            ids,_=train_test_split(ids,train_size=1000,stratify=y[ids],random_state=OUTER_SEED)
        train,held=train_test_split(ids,test_size=.25,stratify=y[ids],random_state=OUTER_SEED)
        packet(folder/'p.zip',modality,metric,x[train],y[train])
        packet(folder/'q.zip',modality,metric,x[held])
        truth=sealed/(case+'.npy')
        np.save(truth,y[held],allow_pickle=False)
        cases[case]=dict(modality=modality,metric=metric,fit_rows=len(train),held_rows=len(held),
                         train_sha256=sha(folder/'p.zip'),input_sha256=sha(folder/'q.zip'),truth_sha256=sha(truth))
    sources={'original':source_identity(a.root/'pinned-source'),'review':source_identity(ROOT)}
    save(out/'protocol.json',dict(preregistration_sha256=sha(PROTOCOL),cases=cases,sources=sources,
         study_sha256=sha(Path(__file__)),fit_limit=FIT_LIMIT,wall_seconds_per_arm=600,api_calls=0))


def run_case(a,case,arm):
    out=a.root/'paired'
    plan=read(out/'protocol.json')
    root=a.root/'pinned-source' if arm=='original' else ROOT
    require(plan['sources'][arm]==source_identity(root) and plan['study_sha256']==sha(Path(__file__))
            and sha(PROTOCOL)==plan['preregistration_sha256'],'Runtime or protocol changed')
    folder=out/case
    require(sha(folder/'p.zip')==plan['cases'][case]['train_sha256']
            and sha(folder/'q.zip')==plan['cases'][case]['input_sha256'],'Task bytes changed')
    output=folder/arm
    output.mkdir()
    physical=output/'fits.jsonl'
    settings=env(a.weights,physical)
    # Production already enforces each admission's declared component cost. This hook only counts fits.
    settings.pop('AB_B1_PER_WORKER_FIT_LIMIT',None)
    settings['PYTHONPATH']=os.pathsep.join(map(str,(root/'scripts/ab_acceptance_guard',root)))
    for name in ('SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_TASK_BUDGET_ID'):
        settings.pop(name,None)
    dest=output/'delivery'/'predictions.csv'
    def call(args,tag):
        with (output/(tag+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=root,env=settings,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=675)
    call(['solve',folder/'p.zip','--prediction-input',folder/'q.zip','--run-dir',output/'run',
          '--output',dest,'--fit-limit',FIT_LIMIT,'--time-limit',600,*(['--self-review'] if arm=='review' else [])],'solve')
    run=output/'run'/'agent'
    res,chosen=read(run/'run_outcome.json'),read(run/'final_selection.json')
    require(res['status']=='completed' and res['formal_test_accessed'] is False and res['new_api_requests']==0,
            'Incomplete delivery or unexpected model use: '+str(output))
    replay=output/'replay'/'predictions.csv'
    call(['predict',folder/'q.zip','--run-dir',output/'run','--output',replay],'predict')
    fits=len(physical.read_text().splitlines())
    require(sha(dest)==sha(replay) and sha(run/chosen['model_file'])==chosen['model_sha256']
            and 0<fits==res['attempted_classifier_fits']==res['completed_classifier_fits']<=FIT_LIMIT,'Delivery or fits disagree')
    choices=read(run/'decisions.json')
    require(all(d['observed_trials']==list(range(1,i+1)) for i,d in enumerate(choices)),'Action used future evidence')
    review=read(run/'policy_review.json') if arm=='review' else {}
    for v in review.get('proposals',[]):
        require(all(e['trial']>max(v['fit_trials']) for e in v['evaluation']),'Proposal used future observations')
    info=dict(fits=fits,api_calls=0,objective=chosen['primary_metric'],model=chosen['model_sha256'],winner=res['members'],
              prediction=sha(dest),first_choice=choices[0]['choice']['members'],generation=review.get('generation',0),
              proposals=[{k:v[k] for k in ('version','parent','weights','proposal_step','fit_trials','evaluation','status')}
                         for v in review.get('proposals',[])],
              changed_first_options=sum(d.get('review_changed_first',False) for d in choices),
              split_hash=identity({k:v for k,v in read(run/'primary_split.json').items()
                                   if k in ('fit','held','row_sha256','label_sha256')}),source=source_identity(root))
    save(output/'sealed.json',info)
    print(case,arm,'sealed',fits,'fits;',info['generation'],'active generation',flush=True)


def run_all(a):
    # The two arms for a given input share data, time and fit limits. Independent runs never share model state.
    for case in CASES:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_case,a,case,arm) for arm in ARMS]
            for future in futures:
                future.result()


def evaluate(a):
    out=a.root/'paired'
    plan=read(out/'protocol.json')
    records={c:{arm:read(out/c/arm/'sealed.json') for arm in ARMS} for c in CASES}
    for c,pair in records.items():
        require(pair['original']['first_choice']==pair['review']['first_choice']
                and pair['original']['split_hash']==pair['review']['split_hash'],'Cold-start or row mismatch')
        for arm,record in pair.items():
            require(record['prediction']==sha(out/c/arm/'delivery'/'predictions.csv') and
                    record['objective']==plan['cases'][c]['metric'],'Prediction or objective changed')
    save(out/'sealed-predictions.json',{c:{arm:r['prediction'] for arm,r in rows.items()} for c,rows in records.items()})
    comparisons={}
    for c,rows in records.items():
        data=plan['cases'][c]
        path=out/'evaluation'/(c+'.npy')
        require(sha(path)==data['truth_sha256'],'Sealed evaluation targets changed')
        y=np.load(path,allow_pickle=False)
        metric=MetricSpec(data['metric'],tuple(np.unique(y).tolist()))
        scores,hit={},{}
        for arm in ARMS:
            with (out/c/arm/'delivery'/'predictions.csv').open() as file:
                predictions=list(csv.DictReader(file))
            require([r['row_id'] for r in predictions]==list(map(str,range(len(y)))),'Evaluation row order changed')
            predicted=np.asarray([r['prediction'] for r in predictions],dtype=y.dtype)
            scores[arm]=metric.evaluate(y,predicted)['score']
            hit[arm]=predicted==y
        comparisons[c]=dict(metric=metric.name,rows=len(y),scores=scores,delta=scores['review']-scores['original'],
              gained=int(sum(hit['review'] & ~hit['original'])),lost=int(sum(hit['original'] & ~hit['review'])),
              original=rows['original'],review=rows['review'])
    delta={c:r['delta'] for c,r in comparisons.items()}
    average=float(np.mean(list(delta.values())))
    families={'tabular':(delta['n17']+delta['n61'])/2,'text':delta['n29'],'image':delta['n43']}
    qualify=average>=.01 and sum(d>0 for d in families.values())>=2 and min(delta.values())>=-.02 and any(
        r['review']['changed_first_options'] for r in comparisons.values())
    save(out/'results.json',dict(cases=comparisons,mean_delta=average,modality_deltas=families,
         qualifies_for_replication=bool(qualify),default='pinned floor',actual_fits=sum(
         r['fits'] for case in records.values() for r in case.values()),api_calls=0,formal_test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','score'])
    p.add_argument('root',type=Path)
    p.add_argument('--sms',type=Path)
    p.add_argument('--weights',type=Path)
    a=p.parse_args()
    a.root=a.root.resolve()
    if a.weights:a.weights=a.weights.resolve()
    {'prepare':prepare,'run':run_all,'score':evaluate}[a.action](a)
