"""Three equally budgeted, anonymous paths; score only after all available predictions are frozen."""
import argparse
from collections import Counter
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

from scripts.run_b_kernel_domain_transfer import read_original
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_planner_lineage import original_traces
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec


PLAN = ROOT/'research/feedback_reflection_1125/protocol.json'
RULE = ROOT/'research/rsi_lineage_1125/promoted_rule.json'
CASES = ('r41','r42','r43','r44')
MODES = ('control','local','flash')


def source():
    return dict(runtime=source_identity(ROOT), plan=sha(PLAN), runner=sha(Path(__file__)), rule=sha(RULE),
                observer=sha(ROOT/'scripts/reflection_study_observer/sitecustomize.py'),
                fit_observer=sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py'))


def memory(prior):
    bins = {b:Counter(improved=0,tied=0,worse=0,tasks=0) for b in ('near','different')}
    digests = {}
    for name,path in original_traces(prior).items():
        decisions, hist = read(path/'decisions.json'), read(path/'history.json')
        digests[name] = identity([sha(path/'decisions.json'),sha(path/'history.json')])
        seen, past = set(), []
        for i,r in enumerate(hist):
            d=decisions[i]
            require(d['observed_trials']==[row['trial'] for row in hist[:i]], 'Historical evidence ordering')
            if r['status']!='completed' or r['split_id']!='primary':
                continue
            if past:
                novelty=d['choice']['evidence']['novelty']
                group='different' if novelty>=.5 else 'near'
                delta=r['search_score']-max(past)
                bins[group]['improved' if delta>1e-12 else 'worse' if delta < -1e-12 else 'tied']+=1
                seen.add(group)
            past.append(r['search_score'])
        for group in seen:
            bins[group]['tasks']+=1
    return bins,digests


def datasets(archive):
    rng=np.random.default_rng(112541)
    x=rng.normal(size=(1000,10))
    y=(np.sin(2.7*x[:,0])*(x[:,1]+.3)+.42*x[:,2]>.19) ^ (rng.random(1000)<.08)
    yield 'r41','tabular','balanced_accuracy',x,y.astype(int)
    texts,labels=zip(*read_original(archive)['amazon'])
    yield 'r42','text','macro_f1',np.asarray(texts),np.asarray(labels)
    rng=np.random.default_rng(112543)
    x=rng.integers(0,7,size=(1000,8))
    y=np.where(x[:,0]<3,x[:,1]//2==x[:,2]//2,x[:,3]+x[:,4]>6) ^ (rng.random(1000)<.1)
    yield 'r43','categorical','macro_f1',np.char.add('c',x.astype(str)),y.astype(int)
    rng=np.random.default_rng(112544)
    y=rng.integers(0,3,800)
    row,col=np.indices((14,14))
    centers=rng.integers(4,10,size=(800,2,1,1))
    dy,dx=row-centers[:,0],col-centers[:,1]
    radius=rng.choice([2.5,4.5],size=(800,1,1))
    axis=np.select([y[:,None,None]==0,y[:,None,None]==1],[dx,dy],default=dx+dy)
    signal=(np.abs(axis)<=1)&(dx**2+dy**2<=radius**2)
    x=np.clip(rng.normal(95,32,size=signal.shape)+rng.uniform(40,95,(800,1,1))*signal,0,255).astype('uint8')
    yield 'r44','image','accuracy',x,y


def prepare(root, prior, archive):
    root.mkdir(parents=True,exist_ok=False)
    (root/'evaluation').mkdir()
    archive_memory,digests=memory(prior)
    save(root/'prior.json',archive_memory)
    specs={}
    for name,kind,metric,x,y in datasets(archive):
        folder=root/name
        folder.mkdir()
        ids=distinct(x,y)
        a,b=train_test_split(ids,test_size=.25,stratify=y[ids],random_state=112546)
        packet(folder/'train.zip',kind,metric,x[a],y[a])
        packet(folder/'input.zip',kind,metric,x[b])
        truth=root/'evaluation'/(name+'.npy')
        np.save(truth,y[b],allow_pickle=False)
        specs[name]=dict(metric=metric,train=len(a),outer=len(b),train_sha=sha(folder/'train.zip'),
                         features_sha=sha(folder/'input.zip'),truth_sha=sha(truth))
    save(root/'protocol.json',dict(source=source(),inputs=specs,history=digests,prior=sha(root/'prior.json')))


def run_one(root, case, mode, weights, ledger=None, key=None):
    protocol=read(root/'protocol.json')
    spec=protocol['inputs'][case]
    folder=root/case
    require(protocol['source']==source() and protocol['prior']==sha(root/'prior.json') and
            spec['train_sha']==sha(folder/'train.zip') and spec['features_sha']==sha(folder/'input.zip'),'Frozen inputs changed')
    work=folder/mode
    work.mkdir()
    settings=env(weights,work/'fits.jsonl')
    settings['PYTHONPATH']=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT)))
    for k in ('AB_B1_PER_WORKER_FIT_LIMIT','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE',
              'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX','REFLECTION_STUDY_LIVE'):
        settings.pop(k,None)
    if mode=='flash':
        settings.update(REFLECTION_STUDY_LIVE='1',SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(ledger),
                        SUPERAGENT_TASK_BUDGET_ID='feedback-'+case+'-1125',SUPERAGENT_PLANNER_KEY_FILE=str(key),
                        NO_PROXY='aigc-api.hkust-gz.edu.cn',no_proxy='aigc-api.hkust-gz.edu.cn')
    run=work/'run'
    output=work/'delivery/predictions.csv'
    replay=work/'replay/predictions.csv'
    for command,args in [('solve',[folder/'train.zip','--prediction-input',folder/'input.zip','--run-dir',run,
            '--output',output,'--fit-limit',16,'--time-limit',600,'--exploration-rule',RULE,'--protected-self-review',
            '--feedback-reflection',mode,'--reflection-history',root/'prior.json']),
            ('predict',[folder/'input.zip','--run-dir',run,'--output',replay])]:
        with (work/(command+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','unified_agent',command,*map(str,args)],cwd=ROOT,env=settings,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=690)
    agent=run/'agent'
    out,final=read(agent/'run_outcome.json'),read(agent/'final_selection.json')
    fits=len((work/'fits.jsonl').read_text().splitlines())
    decisions,hist=read(agent/'decisions.json'),read(agent/'history.json')
    reflection=read(agent/'reflection.json') if (agent/'reflection.json').exists() else None
    entries=[json.loads(s) for s in (agent/'llm_usage.jsonl').read_text().splitlines()]
    calls=[{k:e.get(k) for k in ('event','role','budget_request_id','response_id','served_model','input_tokens',
           'output_tokens','course_cost_rmb','actual_cost_rmb')} for e in entries if e.get('budget_request_id')]
    require(out['status']=='completed' and not out['formal_test_accessed'] and out['training_only'] and final['refit']
            and final['primary_metric']==spec['metric'] and final['training_rows']==spec['train']
            and 0<fits==out['attempted_classifier_fits']==out['completed_classifier_fits']<=16
            and final['model_sha256']==sha(agent/final['model_file']) and sha(output)==sha(replay)
            and len(calls)==out['new_api_requests']<=int(mode=='flash'),'Model, replay or use mismatch')
    require(all(d['observed_trials']==[r['trial'] for r in hist[:i]] for i,d in enumerate(decisions)), 'Future trial access')
    if reflection:
        step=reflection['step']; d=decisions[step-1]
        require(reflection['input_digest']==identity(read(agent/'reflection_input.json')) and
                d['choice']==d['alternatives'][reflection['decision']['option']] and d['source']==mode+' feedback reflection',
                'Advice was not executed on its own state')
    else:
        step=len(decisions)+1
    prefix=[[d['choice']['action'],d['choice']['members'],d['choice']['split']] for d in decisions[:step-1]]
    receipt=dict(case=case,mode=mode,reflection=reflection,prefix=identity(prefix),split=sha(agent/'primary_split.json'),
                 fits=fits,model=final['model_sha256'],prediction=sha(output),winner=out['members'],calls=calls,
                 unknown=out['unresolved_api_requests'],seconds=out['wall_seconds'])
    save(work/'sealed.json',receipt)
    print(case,mode,'fits',fits,'verdict',reflection['decision']['verdict'] if reflection else 'not_triggered',
          'calls',len(calls),'unknown',receipt['unknown'],flush=True)
    return receipt


def local(root, weights):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_one,root,c,m,weights) for c in CASES for m in ('control','local')]
        for future in futures:
            future.result()


def live(root, weights, ledger, key):
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        before={r[0] for r in db.execute('SELECT id FROM requests')}
    for case in CASES:
        rec=run_one(root,case,'flash',weights,ledger,key)
        if rec['unknown']:
            print('Halted new requests after unresolved observed usage',flush=True)
            break
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM requests ORDER BY rowid') if r['id'] not in before]
        ids={r['id'] for r in rows}
        transitions=[dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY seq') if r['request_id'] in ids]
    save(root/'pending_merge.json',dict(team_merge_state='pending_team_handoff',team_total_verified=False,
         requests=rows,transitions=transitions,source='existing provisional B outbox'))


def score(root):
    protocol=read(root/'protocol.json')
    require(protocol['source']==source(),'Frozen implementation changed')
    sealed={(c,m):read(root/c/m/'sealed.json') for c in CASES for m in MODES if (root/c/m/'sealed.json').exists()}
    require(all((c,'local') in sealed and (c,'control') in sealed for c in CASES),'Missing baseline')
    for (c,m),r in sealed.items():
        require(sha(root/c/m/'delivery/predictions.csv')==r['prediction'],'Frozen output changed')
        base=sealed[c,'control']
        require(r['prefix']==base['prefix'] and r['split']==base['split'] and
                (r['reflection'] or {}).get('input_digest')==(base['reflection'] or {}).get('input_digest'),
                'Comparison used different evidence')
    save(root/'all_predictions_frozen.json',{c:{m:sealed[c,m]['prediction'] for m in MODES if (c,m) in sealed} for c in CASES})
    results={};counts={k:Counter() for k in ('flash_vs_local','flash_vs_control','local_vs_control')}
    for c in CASES:
        truth=root/'evaluation'/(c+'.npy')
        spec=protocol['inputs'][c]
        require(sha(truth)==spec['truth_sha'],'Evaluation labels changed')
        y=np.load(truth,allow_pickle=False);metric=MetricSpec(spec['metric'],tuple(np.unique(y).tolist()))
        predictions={};graded={}
        for m in MODES:
            if (c,m) not in sealed:
                continue
            with (root/c/m/'delivery/predictions.csv').open(newline='') as f:
                rows=list(csv.DictReader(f))
            require([r['row_id'] for r in rows]==list(map(str,range(len(y)))),'Row order mismatch')
            p=np.asarray([r['prediction'] for r in rows],dtype=y.dtype);predictions[m]=p
            graded[m]=dict(score=metric.evaluate(y,p)['score'],correct=int((p==y).sum()),rows=len(y),receipt=sealed[c,m])
        comparisons={}
        for left,right in [('flash','local'),('flash','control'),('local','control')]:
            if left not in graded:continue
            label=left+'_vs_'+right
            diff=graded[left]['score']-graded[right]['score']; new=predictions[left]==y; old=predictions[right]==y
            counts[label]['win' if diff>1e-12 else 'loss' if diff< -1e-12 else 'tie']+=1
            comparisons[label]=dict(delta=diff,corrected=int(sum(new & ~old)),spoiled=int(sum(~new & old)))
        results[c]=dict(metric=spec['metric'],arms=graded,comparisons=comparisons)
    usage=[e for r in sealed.values() for e in r['calls']]
    known=[e for e in usage if e['course_cost_rmb'] is not None]
    for row in known:
        require(row['role']=='feedback_reflection' and row['served_model'].casefold()=='deepseek-v4.1-flash' and
                abs((row['input_tokens']+4*row['output_tokens'])/1e6-row['course_cost_rmb'])<1e-10,'Usage pricing mismatch')
    fits=sum(r['fits'] for r in sealed.values())
    require(fits==sum(len(f.read_text().splitlines()) for f in root.glob('*/*/fits.jsonl'))<=192,'Study accounting mismatch')
    save(root/'completion.json',dict(cases=results,comparisons=counts,completed_runs=len(sealed),fits=fits,
        api_calls=len(usage),unknown_calls=len(usage)-len(known),known_rmb=sum(e['course_cost_rmb'] for e in known),
        known_input=sum(e['input_tokens'] for e in known),known_output=sum(e['output_tokens'] for e in known),
        prior=read(root/'prior.json'),source=source(),formal_test_accessed=False))
    print('Completed',len(sealed),'runs,',fits,'fits,',len(usage),'calls',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','local','live','score'))
    parser.add_argument('root',type=Path)
    for k in ('prior','archive','weights','ledger','key-file'):
        parser.add_argument('--'+k,type=Path)
    a=parser.parse_args();root=a.root.resolve()
    if a.stage=='prepare':prepare(root,a.prior.resolve(),a.archive.resolve())
    elif a.stage=='local':local(root,a.weights.resolve())
    elif a.stage=='live':live(root,a.weights.resolve(),a.ledger.resolve(),a.key_file.resolve())
    else:score(root)
