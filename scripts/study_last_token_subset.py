"""Compare the same autonomous search with and without an added last-token resource on training groups."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT,env,read,require,sha,text_packet
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import bank_path,encode,installed
from superagent.paired_encoder import AVAILABLE_ENCODERS, bank_name, text_key
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN=ROOT/'research/paired_encoder_search_1127/last_token_subset_protocol.json'


def take_groups(groups,indices,wanted):
    ids=set(groups[indices])
    ordered=sorted(ids,key=lambda s:hashlib.sha256(('last-token subset v1/'+s).encode()).digest())
    chosen=[]
    total=0
    for g in ordered:
        chosen.append(g)
        total+=np.count_nonzero(groups[indices]==g)
        if total>=wanted:
            break
    return indices[np.isin(groups[indices],chosen)]


def prepare(root):
    out=root/'last-token-subset'
    out.mkdir(exist_ok=False)
    require(not read(root/'last-token-timing.json')['use_full_training'],'Expected registered subset condition')
    agent=root/'raw/race/run/agent'
    path=agent/'refit_data.joblib'
    data=joblib.load(path)
    x,y=data['X_train'],np.asarray(data['y_train'])
    g=sample_groups(TrainingTask(data['kind'],x,y),'shared_text')
    split=read(agent/'primary_split.json')
    fit=take_groups(g,np.asarray(split['fit']),768)
    held=take_groups(g,np.asarray(split['held']),256)
    require(not(set(g[fit]) & set(g[held])) and len(fit)>=768 and len(held)>=256,'Invalid subset')
    text_packet(out/'anonymous.zip',[(*x[i],int(y[i])) for i in fit],True)
    text_packet(out/'features.zip',[x[i] for i in held],False)
    x_train=sorted({s for i in fit for s in x[i]})
    joblib.dump(x_train,out/'training_texts.joblib')
    lock(out/'inputs.json',dict(protocol=sha(PLAN),source=sha(path),train=fit.tolist(),held=held.tolist(),
        fit_groups=len(set(g[fit])),held_groups=len(set(g[held])),train_fingerprint=fingerprint(fit),
        held_fingerprint=fingerprint(held),features=sha(out/'features.zip'),task=sha(out/'anonymous.zip'),
        train_texts=len(x_train),train_text_sha=sha(out/'training_texts.joblib')))
    cache=out/'features_train_only'
    cache.mkdir()
    keys={text_key(s) for s in x_train}
    activate_resource_directory(root/'discovery-library')
    paths=[root/'feature_bank'/bank_name(v) for v in AVAILABLE_ENCODERS]
    paths.extend(bank_path(root/'feature_bank',spec) for spec in installed())
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path,allow_pickle=False) as old:
            old_keys=old['keys']
            use=np.isin(old_keys,list(keys))
            require(use.sum()==len(keys),'Missing inherited train-only vectors')
            np.savez(cache/path.name,keys=old_keys[use],vectors=old['vectors'][use])
    print(len(fit),len(held),len(x_train),'prepared',flush=True)


def warm(root):
    out=root/'last-token-subset'
    register=read(out/'inputs.json')
    source=out/'training_texts.joblib'
    require(sha(source)==register['train_text_sha'],'Training texts changed')
    texts=joblib.load(source)
    activate_resource_directory(root/'last-token-library')
    specs=[s for s in installed() if s['pooling']=='last']
    require(len(specs)==1,'Need one added declarative candidate')
    spec=specs[0]
    target=bank_path(out/'features_train_only',spec)
    require(not target.exists(),'Train-only vectors already exist')
    os.environ['SUPERAGENT_ENCODING_DEVICE_LOG']=str(out/'device.jsonl')
    t=time.perf_counter()
    values,count=encode(spec,texts,return_tokens=True)
    duration=time.perf_counter()-t
    keys=[text_key(s) for s in texts]
    with NamedTemporaryFile(dir=target.parent,delete=False) as file:
        tmp=Path(file.name)
        np.savez(file,keys=keys,vectors=values)
    try:
        os.link(tmp,target)
    finally:
        tmp.unlink(missing_ok=True)
    lock(out/'warming.json',dict(input=sha(out/'inputs.json'),asset=spec['signature'],rows=len(texts),tokens=count,
                                seconds=duration,bank=sha(target),model_api_requests=0))
    print('training cache',len(texts),count,round(duration,2),flush=True)


def solve(root,arm):
    out=root/'last-token-subset'
    reg=read(out/'inputs.json')
    source,features=out/'anonymous.zip',out/'features.zip'
    require(sha(source)==reg['task'] and sha(features)==reg['features'],'Subset inputs changed')
    if arm=='expanded':
        require(read(out/'warming.json')['input']==sha(out/'inputs.json'),'No frozen subset vectors')
    weights=root/('last-token-library' if arm=='expanded' else 'discovery-library')
    stage=out/arm
    stage.mkdir(exist_ok=False)
    physical=stage/'physical.jsonl'
    environment=env(weights,physical)
    for key in tuple(environment):
        if key.startswith('SUPERAGENT_') and (key.endswith('_WEIGHTS') or key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_WEIGHTS_DIR')):
            environment.pop(key)
    environment['SUPERAGENT_PAIRED_FEATURE_CACHE']=str(out/'features_train_only')
    environment['SUPERAGENT_ENCODING_DEVICE_LOG']=str(stage/'hardware.jsonl')
    run=stage/'run'
    prediction=stage/'prediction.csv'
    replay=stage/'replay.csv'
    lock(stage/'launch.json',dict(runtime=source_identity(ROOT),protocol=sha(PLAN),input=sha(out/'inputs.json'),
         fit_limit=28,seconds=1800,model_hint=None,strategy_hint=None,resource_root=weights.name))
    for tag,cmd in [('solve',['solve',source,'--weights-dir',weights,'--fit-limit','28','--time-limit','1800',
                               '--prediction-input',features,'--run-dir',run,'--output',prediction]),
                    ('predict',['predict',features,'--weights-dir',weights,'--run-dir',run,'--output',replay])]:
        if tag=='predict':
            environment.pop('SUPERAGENT_PAIRED_FEATURE_CACHE',None)
        with (stage/(tag+'.log')).open('x') as stream:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,cmd)],cwd=ROOT,env=environment,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=2050,check=True)
    agent=run/'agent'
    finished=read(agent/'run_outcome.json')
    selection=read(agent/'final_selection.json')
    fits=len(physical.read_text().splitlines())
    require(finished['status']=='completed' and finished['attempted_classifier_fits']==fits<=28 and
            finished['new_api_requests']==0 and selection['refit'] and selection['training_rows']==len(reg['train']),
            'Autonomous run did not complete within agreed bounds')
    require(sha(prediction)==sha(replay),'Uncached prediction differs')
    steps=read(agent/'decisions.json')
    history=read(agent/'history.json')
    offered=sorted({n for a in steps[0]['alternatives'] for n in a['members'] if n.startswith('local_sentence_')})
    lock(stage/'sealed.json',dict(protocol=sha(stage/'launch.json'),file=sha(prediction),model=selection['model_sha256'],
        selected=steps[-1]['choice']['members'],offered_local=offered,menu_size=len(steps[0]['alternatives']),
        explored_local=[dict(member=x['candidate'],fold=x['split_id'],score=x['search_score']) for x in history
                        if x['candidate'] in offered and x['status']=='completed'],fits=fits,independent_reload=True,
        estimate=selection['selection_score'],api=0))
    print(arm,fits,steps[-1]['choice']['members'],flush=True)


def score(root):
    out=root/'last-token-subset'
    samples=read(out/'inputs.json')
    results={}
    predictions={}
    cases=('original','expanded')
    seals={k:read(out/k/'sealed.json') for k in cases}
    require(len(set(seals['expanded']['offered_local'])-set(seals['original']['offered_local']))==1,
            'Expected exactly one added executable method')
    lock(out/'evaluation_lock.json',dict(inputs=sha(out/'inputs.json'),
                                        seals={k:sha(out/k/'sealed.json') for k in cases}))
    source=root/'raw/race/run/agent/refit_data.joblib'
    require(sha(source)==samples['source'],'Evaluation source changed')
    truth=np.asarray(joblib.load(source)['y_train'])[samples['held']]
    for k in cases:
        path=out/k/'prediction.csv'
        require(sha(path)==seals[k]['file'],'Sealed predictions changed')
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        require([int(x['row_id']) for x in rows]==list(range(len(truth))),'Held order mismatch')
        pred=np.asarray([int(x['prediction']) for x in rows])
        predictions[k]=pred==truth
        results[k]=dict(correct=int(np.sum(pred==truth)),rows=len(truth),**seals[k])
    base,exp=[predictions[k] for k in cases]
    extra=next(iter(set(seals['expanded']['offered_local'])-set(seals['original']['offered_local'])))
    result=dict(lock=sha(out/'evaluation_lock.json'),results=results,additional_member=extra,
        additional_member_tried=any(r['member']==extra for r in seals['expanded']['explored_local']),
        gained=int((exp&~base).sum()),lost=int((base&~exp).sum()),training_only=True,course_validation_opened=False,
        official_test_opened=False,H2_opened=False,model_api_requests=0)
    lock(out/'results.json',result)
    print({k:(v['correct'],v['rows'],v['fits'],v['selected']) for k,v in results.items()},result['gained'],result['lost'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','warm','solve','score'))
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--arm',choices=('original','expanded'))
    a=p.parse_args()
    if a.mode=='solve':
        solve(a.root,a.arm)
    else:
        {'prepare':prepare,'warm':warm,'score':score}[a.mode](a.root)
