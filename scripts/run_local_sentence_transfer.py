"""Execute the ordinary controller on an anonymous single-text task; separately test the portable plugin."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from scripts.build_external_tasks import RAW, split_rows
from scripts.run_bq_01_portfolios import pack
from scripts.unified_released_study import ROOT,env,read,require,save,sha
from superagent.external_assets import activate_resource_directory, model_resources, verify_resources
from superagent.local_sentence import candidates
from superagent.splits import normalized_text


PLAN=ROOT/'research/paired_encoder_search_1127/discovery_text_protocol.json'


def samples(source):
    require(sha(source)==RAW['sms_spam']['sha256'].lower(),'Unexpected public archive')
    rows={}
    with zipfile.ZipFile(source) as z:
        for line in z.read('SMSSpamCollection').decode('utf-8').splitlines():
            if '\t' not in line:
                continue
            y,x=line.split('\t',1)
            if x.strip():
                rows.setdefault(x.strip(),y.strip())
    texts=list(rows)
    universe,_=split_rows(np.array(list(rows.values())))
    groups={}
    for i in universe:
        s=texts[i]
        groups.setdefault(normalized_text(s),(s,rows[s]))
    ordered=sorted(groups,key=lambda v:hashlib.sha256(('modular text smoke v1/'+v).encode()).hexdigest())
    return [groups[k] for k in ordered[:640]], [groups[k] for k in ordered[640:800]]


def run(source,root,weights):
    out=root/'discovered-single-text'
    out.mkdir(exist_ok=False)
    train,audit=samples(source)
    tx,ty=map(list,zip(*train))
    qx,qy=map(list,zip(*audit))
    require(not {normalized_text(s) for s in tx}&{normalized_text(s) for s in qx},'Dependency groups overlap')
    task,features=out/'dataset.zip',out/'features.zip'
    pack(task,'single_text',tx,ty,'macro_f1','train')
    pack(features,'single_text',qx,None,'macro_f1','test')
    save(out/'lock.json',dict(protocol=sha(PLAN),source=sha(source),task=sha(task),features=sha(features),
                              training=len(tx),external=len(qx)))
    logs=out/'classifier.jsonl'
    environment=env(weights,logs)
    for k in tuple(environment):
        if k.startswith('SUPERAGENT_') and (k.endswith('_WEIGHTS') or k in ('SUPERAGENT_PAIRED_FEATURE_CACHE','SUPERAGENT_WEIGHTS_DIR','SUPERAGENT_PLANNER_KEY_FILE')):
            environment.pop(k)
    def call(args,log):
        with (out/log).open('x') as stream:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=ROOT,env=environment,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1350)
    fresh=out/'predictions.csv'
    replay=out/'reloaded.csv'
    run_dir=out/'run'
    call(['solve',task,'--weights-dir',weights,'--fit-limit','18','--time-limit','1200','--run-dir',run_dir,
          '--prediction-input',features,'--output',fresh],'solve.log')
    call(['predict',features,'--weights-dir',weights,'--run-dir',run_dir,'--output',replay],'reload.log')
    require(sha(replay)==sha(fresh),'Predictions changed in a fresh process')
    agent=run_dir/'agent'
    history=read(agent/'history.json')
    steps=read(agent/'decisions.json')
    chosen=steps[-1]['choice']['members']
    total=len(logs.read_text().splitlines())
    result=read(agent/'run_outcome.json')
    require(result['status']=='completed' and result['attempted_classifier_fits']==total<=18 and result['new_api_requests']==0,
            'Run did not complete in the registered budget')
    known={n for a in steps[0]['alternatives'] for n in a['members'] if n.startswith('local_sentence_')}
    with fresh.open(newline='') as stream:
        predictions=list(csv.DictReader(stream))
    require([int(r['row_id']) for r in predictions]==list(range(len(qy))),'Row mismatch')
    labels=[r['prediction'] for r in predictions]
    save(out/'autonomous.json',dict(protocol=sha(PLAN),initial_options=len(steps[0]['alternatives']),offered=sorted(known),
         explored=[r['candidate'] for r in history if r['candidate'] in known],selected=chosen,rows=len(qy),
         correct=int(sum(a==b for a,b in zip(labels,qy))),accuracy=accuracy_score(qy,labels),
         macro_f1=f1_score(qy,labels,average='macro'),actual_classifier_fits=total,prediction_sha256=sha(replay),
         independent_reload=True,model_api_requests=0))
    activate_resource_directory(weights)
    options=candidates('single_text')
    def footprint(candidate):
        s=candidate.fit_predict.keywords['spec']
        return (s['files']['model.safetensors']['bytes'],s['signature'])
    chosen=min(options,key=footprint)
    model,expected=chosen.fit_predict(tx,ty,qx)
    resources,_=model_resources(model)
    require(len(resources)==1 and resources[0]['environment_variable'].startswith('SUPERAGENT_SENTENCE_ASSET_'),
            'Exported the wrong model family')
    verify_resources(resources)
    path=out/'compatibility.joblib'
    joblib.dump(model,path)
    inputs=out/'compatibility_inputs.joblib'
    joblib.dump(qx,inputs)
    os.environ.pop('SUPERAGENT_PAIRED_FEATURE_CACHE',None)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        asset=resources[0]
        old=Path(os.environ[asset['environment_variable']])
        renamed=Path(temp)/'opaque_resource'
        for relative in asset['files']:
            target=renamed/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            os.link(old/relative,target)
        child=dict(os.environ)
        for key in tuple(child):
            if key.startswith('SUPERAGENT_') and (key.endswith('_WEIGHTS') or key=='SUPERAGENT_WEIGHTS_DIR'):
                child.pop(key)
        proof=out/'compatibility_predictions.joblib'
        subprocess.run([sys.executable,'-m','scripts.run_local_sentence_transfer','reload','--model',str(path),
                        '--inputs',str(inputs),'--weights',str(Path(temp)),'--destination',str(proof)],cwd=ROOT,
                       env=child,check=True,stdin=subprocess.DEVNULL)
        labels,prob=joblib.load(proof)
        require(np.array_equal(expected,labels) and np.allclose(model.predict_proba(qx),prob,rtol=0,atol=1e-7),
                'Relocated exported encoder prediction differs')
    save(out/'compatibility.json',dict(model=chosen.family,revision=asset['revision'],resource_count=len(resources),
        records=len(qy),actual_classifier_fits=1,model_api_requests=0,relocated_opaque_directory=True,
        exact_labels=True,max_probability_error=float(np.max(np.abs(model.predict_proba(qx)-prob)))))
    print(read(out/'autonomous.json'),read(out/'compatibility.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('run','reload'))
    for k in ('source','root','weights','model','inputs','destination'):
        p.add_argument('--'+k,type=lambda x:Path(x).resolve())
    args=p.parse_args()
    if args.operation=='run':
        run(args.source,args.root,args.weights)
    else:
        activate_resource_directory(args.weights)
        model=joblib.load(args.model)
        verify_resources(model_resources(model)[0])
        rows=joblib.load(args.inputs)
        joblib.dump((model.predict(rows), model.predict_proba(rows)),args.destination)
