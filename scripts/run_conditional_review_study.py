"""Cold, budget-matched comparison of original, early and contingent review on fresh development splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.audit_interrupted_reflection import freeze
from scripts.run_b_kernel_domain_transfer import read_original
from scripts.run_feedback_reflection_study import RULE
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_feedback_reflection_study import independent_score
from superagent.experiment_policy import identity


PLAN=ROOT/'research/conditional_review_1125/protocol.json'
CASES=('s51','s52','s53','s54')
MODES=('control','flash','conditional-flash')


def source():
    files=[PLAN,RULE,Path(__file__),ROOT/'scripts/audit_interrupted_reflection.py',
           ROOT/'scripts/reflection_study_observer/sitecustomize.py',ROOT/'scripts/ab_acceptance_guard/sitecustomize.py']
    return dict(runtime=source_identity(ROOT),files={p.name if p.name!='sitecustomize.py' else p.parent.name:sha(p) for p in files})


def datasets(archive):
    rng=np.random.default_rng(26051)
    x=rng.normal(size=(1000,12))
    x[:,4]=.7*x[:,0]+.3*x[:,4]
    y=(x[:,0]**2+.7*x[:,1]*x[:,2]+.45*np.sin(3*x[:,3])>1.15) ^ (rng.random(1000)<.1)
    yield 's51','tabular','balanced_accuracy',x,y.astype(int)
    sentences,labels=zip(*read_original(archive)['imdb'])
    yield 's52','text','macro_f1',np.asarray(sentences),np.asarray(labels)
    rng=np.random.default_rng(26053)
    x=rng.integers(0,6,size=(1050,7))
    y=np.where(x[:,0]<3,(x[:,1]+(x[:,2]==x[:,3]))%3,(x[:,4]//2+x[:,5]//2)%3)
    noise=rng.random(len(y))<.15
    y[noise]=rng.integers(0,3,size=noise.sum())
    yield 's53','categorical','macro_f1',np.char.add('v',x.astype(str)),y
    rng=np.random.default_rng(26054)
    y=rng.integers(0,3,720)
    row,col=np.indices((12,12))
    centers=rng.uniform(3,9,(720,2,1,1))
    dy,dx=row-centers[:,0],col-centers[:,1]
    axis=np.choose(y[:,None,None],[dx,dy,(dx-dy)/1.4142])
    width=rng.uniform(2,4,size=(720,1,1))
    signal=np.cos(1.55*axis)*np.exp(-(dx**2+dy**2)/(2*width**2))
    pixels=np.clip(rng.normal(120,32,size=signal.shape)+55*signal,0,255).astype('uint8')
    yield 's54','image','accuracy',pixels,y


def prepare(root,prior,archive):
    root.mkdir(parents=True,exist_ok=False)
    (root/'evaluation').mkdir()
    save(root/'prior.json',read(prior))
    inputs={}
    for case,kind,metric,x,y in datasets(archive):
        folder=root/case
        folder.mkdir()
        idx=distinct(x,y)
        fit,out=train_test_split(idx,test_size=.25,stratify=y[idx],random_state=112557)
        packet(folder/'train.zip',kind,metric,x[fit],y[fit])
        packet(folder/'input.zip',kind,metric,x[out])
        truth=root/'evaluation'/(case+'.npy')
        np.save(truth,y[out],allow_pickle=False)
        inputs[case]=dict(metric=metric,train=len(fit),outer=len(out),train_sha=sha(folder/'train.zip'),
                          input_sha=sha(folder/'input.zip'),truth_sha=sha(truth))
    save(root/'protocol.json',dict(source=source(),inputs=inputs,prior=sha(root/'prior.json')))


def run_one(args,case,mode):
    root=args.root
    folder=root/case
    protocol=read(root/'protocol.json')
    spec=protocol['inputs'][case]
    require(protocol['source']==source() and protocol['prior']==sha(root/'prior.json')
            and spec['train_sha']==sha(folder/'train.zip') and spec['input_sha']==sha(folder/'input.zip'),'Frozen inputs changed')
    work=folder/mode
    if (work/'sealed.json').exists():
        return read(work/'sealed.json')
    work.mkdir(exist_ok=True)
    settings=env(args.weights,work/'fits.jsonl')
    settings['PYTHONPATH']=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT)))
    for k in ('AB_B1_PER_WORKER_FIT_LIMIT','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE',
              'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX','REFLECTION_STUDY_LIVE',
              'SUPERAGENT_RESEARCH_PAIRED_REFLECTION'):
        settings.pop(k,None)
    if mode!='control':
        settings.update(REFLECTION_STUDY_LIVE='1',SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(args.ledger),
                        SUPERAGENT_TASK_BUDGET_ID='conditional-review-'+case,SUPERAGENT_PLANNER_KEY_FILE=str(args.key_file),
                        SUPERAGENT_RESEARCH_PAIRED_REFLECTION='1',NO_PROXY='aigc-api.hkust-gz.edu.cn',no_proxy='aigc-api.hkust-gz.edu.cn')
    run=work/'run'
    output,replay=(work/x/'predictions.csv' for x in ('delivery','replay'))
    commands=[('solve',output,[folder/'train.zip','--prediction-input',folder/'input.zip','--run-dir',run,
               '--output',output,'--fit-limit',16,'--time-limit',600,'--exploration-rule',RULE,'--protected-self-review',
               '--feedback-reflection',mode,'--reflection-history',root/'prior.json']),
              ('predict',replay,[folder/'input.zip','--run-dir',run,'--output',replay])]
    for command,dest,rest in commands:
        if dest.exists():
            continue
        with (work/(command+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','unified_agent',command,*map(str,rest)],cwd=ROOT,env=settings,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=690)
    agent=run/'agent'
    outcome,final=read(agent/'run_outcome.json'),read(agent/'final_selection.json')
    decisions,history=read(agent/'decisions.json'),read(agent/'history.json')
    calls=[json.loads(s) for s in (agent/'llm_usage.jsonl').read_text().splitlines()]
    calls=[e for e in calls if e.get('budget_request_id')]
    fits=len((work/'fits.jsonl').read_text().splitlines())
    require(outcome['status']=='completed' and outcome['training_only'] and not outcome['formal_test_accessed'] and final['refit']
            and final['training_rows']==spec['train'] and final['primary_metric']==spec['metric']
            and 0<fits==outcome['attempted_classifier_fits']==outcome['completed_classifier_fits']<=16
            and sha(output)==sha(replay) and sha(agent/final['model_file'])==final['model_sha256']
            and len(calls)==outcome['new_api_requests']<=int(mode!='control'),'Cold run or accounting mismatch')
    require(all(d['observed_trials']==[r['trial'] for r in history[:i]] for i,d in enumerate(decisions)), 'Future trial access')
    trace=[[d['choice'][k] for k in ('action','members','split')] for d in decisions]
    evidence=read(agent/'conditional_review.json') if mode=='conditional-flash' else read(agent/'reflection.json') if (agent/'reflection.json').exists() else {}
    if mode=='conditional-flash':
        require(not (agent/'reflection.json').exists() and evidence['delivered_sha256']==final['model_sha256'],'Unverified conditional execution')
        if evidence.get('input_digest'):
            require(evidence['input_digest']==identity(read(agent/'conditional_review_input.json')),'Changed reviewer input')
    receipt=dict(fits=fits,model=final['model_sha256'],prediction=sha(output),review=evidence,calls=calls,scope='conditional-review-'+case,
                 unknown=outcome['unresolved_api_requests'],split=sha(agent/'primary_split.json'),pre_review_search=identity(trace),
                 first_three=identity(trace[:3]),completed_trials=sum(r['status']=='completed' for r in history))
    save(work/'sealed.json',receipt)
    print(case,mode,'fits',fits,'calls',len(calls),'review',evidence.get('status'),flush=True)
    return receipt


def paid(args):
    for i,c in enumerate(CASES):
        for mode in (MODES[1:] if i%2==0 else MODES[:0:-1]):
            with sqlite3.connect(args.ledger.as_uri()+'?mode=ro',uri=True) as db:
                if db.execute("SELECT count(*) FROM requests WHERE status!='known'").fetchone()[0]:
                    print('New paid dispatch halted on unresolved ledger usage',flush=True)
                    return
            if run_one(args,c,mode)['unknown']:
                return


def checkpoints(args):
    root=args.root
    available={case:{mode:read(root/case/mode/'sealed.json')['prediction'] for mode in MODES
                      if (root/case/mode/'sealed.json').exists()} for case in CASES}
    require(all('control' in modes for modes in available.values()),'Missing local control')
    save(root/'all_predictions_frozen.json',available)
    out=root/'early'
    out.mkdir(exist_ok=True)
    freeze(root,out,(4,),completed_only=True)


def get_predictions(path,y,digest):
    require(sha(path)==digest,'Changed predictions')
    with path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    require([r['row_id'] for r in rows]==list(map(str,range(len(y)))),'Prediction row order differs')
    return np.asarray([r['prediction'] for r in rows],dtype=y.dtype)


def evaluate(args):
    root=args.root
    protocol=read(root/'protocol.json')
    require(protocol['source']==source(),'Frozen source changed')
    frozen=read(root/'all_predictions_frozen.json')
    early=read(root/'early/frozen.json')
    cases={}
    totals=dict(fits=0,calls=0,input=0,output=0,unknown=0)
    all_requests=set()
    for case,modes in frozen.items():
        spec=protocol['inputs'][case]
        truth=root/'evaluation'/(case+'.npy')
        require(sha(truth)==spec['truth_sha'],'Changed held-out labels')
        y=np.load(truth,allow_pickle=False)
        arms,paired={},{}
        baseline=read(root/case/'control/sealed.json')
        for mode,pred_hash in modes.items():
            work=root/case/mode
            receipt=read(work/'sealed.json')
            require(receipt['split']==baseline['split'] and receipt['first_three']==baseline['first_three'],'Different initial states')
            if mode=='conditional-flash':
                require(baseline['pre_review_search']==receipt['pre_review_search'],'Premature interruption of exploration')
            record=early[case][mode]['4']
            preds={'fourth':get_predictions(root/'early'/record['file'],y,record['digest']),
                   'final':get_predictions(work/'delivery/predictions.csv',y,pred_hash)}
            paired[mode]=preds
            totals['fits']+=receipt['fits']
            usage=[]
            for e in receipt['calls']:
                key=e['budget_request_id']
                require(key not in all_requests,'Repeated request')
                all_requests.add(key)
                with sqlite3.connect(args.ledger.as_uri()+'?mode=ro',uri=True) as db:
                    scope,status,units,payload=db.execute('SELECT scope,status,course_units,payload FROM requests WHERE id=?',(key,)).fetchone()
                require(scope==receipt['scope'] and json.loads(payload)['paired_reflection_arm']==mode,'Wrong accounting scope')
                totals['calls']+=1
                if status=='known':
                    require(e['role']=='feedback_reflection' and e['served_model'].casefold()=='deepseek-v4.1-flash'
                            and units==1000*(e['input_tokens']+4*e['output_tokens']),'Unexpected model or pricing')
                    totals['input']+=e['input_tokens']
                    totals['output']+=e['output_tokens']
                else:
                    totals['unknown']+=1
                usage.append({k:e.get(k) for k in ('served_model','input_tokens','output_tokens','course_cost_rmb')})
            arms[mode]=dict(fits=receipt['fits'],examined_trials=min(4,receipt['completed_trials']),usage=usage,
                review=receipt['review'],scores={key:independent_score(spec['metric'],y,p) for key,p in preds.items()},
                early_sha=record['digest'],final_sha=pred_hash)
        diffs={}
        if 'conditional-flash' in arms:
            for other in ('control','flash'):
                if other not in arms:
                    continue
                diffs[other]={}
                for checkpoint in ('fourth','final'):
                    new,old=paired['conditional-flash'][checkpoint],paired[other][checkpoint]
                    diffs[other][checkpoint]=dict(delta=arms['conditional-flash']['scores'][checkpoint]-arms[other]['scores'][checkpoint],
                        changed=int(sum(new!=old)),corrected=int(sum((new==y)&(old!=y))),spoiled=int(sum((new!=y)&(old==y))))
        cases[case]=dict(metric=spec['metric'],rows=len(y),arms=arms,conditional_vs=diffs)
    physical=sum(len(p.read_text().splitlines()) for p in root.glob('*/*/fits.jsonl'))
    require(totals['fits']==physical<=192 and totals['calls']<=8,'Unmatched physical work')
    with sqlite3.connect(args.ledger.as_uri()+'?mode=ro',uri=True) as db:
        ids={r[0] for c in CASES for r in db.execute('SELECT id FROM requests WHERE scope=?',('conditional-review-'+c,))}
        require(ids==all_requests,'Missing study request')
    if totals['unknown']==0:
        totals['course_rmb']=(totals['input']+4*totals['output'])/1e6
    comparable=all(len(v['arms'])==3 and v['arms']['flash']['review'].get('status')=='valid'
                   and v['arms']['conditional-flash']['review'].get('status')!='retained_after_error' for v in cases.values())
    passed=comparable and all(c['conditional_vs'][o]['final']['delta']>=-1e-12 for c in cases.values() for o in ('control','flash')) and all(
        c['conditional_vs']['control']['fourth']['delta']>=-1e-12 for c in cases.values()) and any(
        c['conditional_vs']['flash'][k]['delta']>1e-12 for c in cases.values() for k in ('fourth','final'))
    report=dict(protocol_sha256=sha(root/'protocol.json'),source=protocol['source'],cases=cases,totals=totals,
                matched_complete=comparable,meets_preregistered_criterion=passed,formal_test_accessed=False,team_ledger_merge='pending')
    save(root/'completion.json',report)
    save(args.output,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','control','paid','checkpoints','score'))
    parser.add_argument('root',type=lambda x:Path(x).resolve())
    for k in ('prior','archive','weights','ledger','key-file','output'):
        parser.add_argument('--'+k,type=lambda x:Path(x).resolve())
    a=parser.parse_args()
    if a.stage=='prepare':
        prepare(a.root,a.prior,a.archive)
    elif a.stage=='control':
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda c:run_one(a,c,'control'),CASES))
    elif a.stage=='paid':
        paid(a)
    elif a.stage=='checkpoints':
        if os.environ.get('AB_B1_OFFLINE')=='1':
            checkpoints(a)
        else:
            fitlog=a.root/'unexpected_checkpoint_fits.jsonl'
            subprocess.run([sys.executable,'-m','scripts.run_conditional_review_study','checkpoints',str(a.root),
                            '--weights',str(a.weights)],cwd=ROOT,env=env(a.weights,fitlog),check=True)
            require(not fitlog.exists() or not fitlog.read_text().strip(),'Checkpoint replay trained a model')
    else:
        evaluate(a)
