"""Three cold-start arms with independently frozen predictions, physical fits and campus settlements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.run_conditional_review_study import get_predictions
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_conditional_study_receipts import score_from_counts
from superagent.experiment_policy import identity
from superagent.predict import create_predictions
from superagent.proposal_value import MODES, DELIVERED_MODES, PAID_MODES


PLAN=ROOT/'research/proposal_value_1126/protocol.json'
DELIVERED_PLAN=PLAN.with_name('delivered_protocol.json')
CASES=('p81','p82','p83','p84')
DELIVERED_CASES=('p85','p86')
ARCHIVE='985e501fbe891f050e2134be6d4cdb79e955435e240ea8f238b3fdb92ed839aa'


def source(delivered=False):
    files=[DELIVERED_PLAN if delivered else PLAN,Path(__file__),ROOT/'scripts/reflection_study_observer/sitecustomize.py',
           ROOT/'scripts/ab_acceptance_guard/sitecustomize.py',ROOT/'scripts/run_planner_self_review.py',
           ROOT/'scripts/verify_conditional_study_receipts.py']
    return dict(runtime=source_identity(ROOT),audit={str(p.relative_to(ROOT)):sha(p) for p in files})


def datasets(archive):
    r=np.random.default_rng(261181)
    x=r.normal(size=(760,13))
    y=(np.where(x[:,0]>.15,.9*x[:,1]-.4*x[:,2],x[:,1]*x[:,3]+.6*np.cos(2*x[:,4]))>.2)^(r.random(len(x))<.19)
    yield 'p81','tabular','balanced_accuracy',x,y.astype(int)
    r=np.random.default_rng(261182)
    x=r.integers(0,7,size=(820,9))
    y=np.where(x[:,0]%2,(x[:,1]//3+x[:,2]%2)%3,(x[:,3]+(x[:,4]==x[:,5]))%3)
    noise=r.random(len(x))<.16
    y[noise]=r.integers(0,3,size=noise.sum())
    yield 'p82','categorical','macro_f1',np.char.add('v',x.astype(str)),y
    r=np.random.default_rng(261183)
    y=r.integers(0,3,640)
    iy,ix=np.indices((14,14))
    cy,cx=r.uniform(4,10,size=(2,len(y),1,1))
    dy,dx=iy-cy,ix-cx
    radius=r.uniform(2.3,3.8,size=(len(y),1,1))
    radial=np.sqrt(dx**2+dy**2)
    signals=[np.abs(radial-radius),np.minimum(abs(dx),abs(dy)),np.minimum(abs(dx-dy),abs(dx+dy))/.7071]
    distance=np.choose(y[:,None,None],signals)
    img=r.normal(100,27,(len(y),14,14))+55*np.exp(-distance**2/.95)*np.exp(-radial**2/65)
    yield 'p83','image','accuracy',np.clip(img,0,255).astype('uint8'),y
    require(sha(archive)==ARCHIVE,'Unexpected public corpus bytes')
    texts,labels=[],[]
    with tarfile.open(archive,'r:gz') as pack:
        for label,basename in enumerate(('plot.tok.gt9.5000','quote.tok.gt9.5000')):
            member=next(m for m in pack.getmembers() if Path(m.name).name==basename and m.isfile())
            lines=pack.extractfile(member).read().decode('latin1').splitlines()
            texts.extend(lines)
            labels.extend([label]*len(lines))
    x,y=np.asarray(texts),np.asarray(labels)
    idx=distinct(x,y)
    subset,_=train_test_split(idx,train_size=960,random_state=261184,stratify=y[idx])
    yield 'p84','text','macro_f1',x[subset],y[subset]


def delivered_datasets():
    r=np.random.default_rng(261185)
    x=r.normal(size=(560,11))
    radius=x[:,0]**2+.8*x[:,1]**2
    signal=np.where(x[:,2]>-.2,1.5-radius,.8*x[:,3]-.65*x[:,4]+.45*x[:,1])
    y=(signal+r.normal(0,.64,len(x))>0)^(r.random(len(x))<.08)
    yield 'p85','tabular','balanced_accuracy',x,y.astype(int)
    r=np.random.default_rng(261186)
    y=r.integers(0,3,480)
    iy,ix=np.indices((16,16))
    cy,cx=r.uniform(5,10,size=(2,len(y),1,1))
    angle=np.asarray([0.,np.pi/3,2*np.pi/3])[y,None,None]+r.normal(0,.19,(len(y),1,1))
    length=r.uniform(1.8,3.3,(len(y),1,1))
    dy,dx=length*np.sin(angle),length*np.cos(angle)
    spots=sum(np.exp(-((iy-cy-s*dy)**2+(ix-cx-s*dx)**2)/3.3) for s in (-1,1))
    image=r.normal(95,28,(len(y),16,16))+r.uniform(56,96,(len(y),1,1))*spots
    yield 'p86','image','accuracy',np.clip(image,0,255).astype('uint8'),y


def prepare(a,smoke=False):
    a.root.mkdir(parents=True,exist_ok=False)
    (a.root/'evaluation').mkdir()
    inputs={}
    if smoke:
        rng=np.random.default_rng(3821)
        x=rng.normal(size=(130,6))
        cases=[('smoke','tabular','accuracy',x,(x[:,1]+rng.normal(size=len(x))>0).astype(int))]
    else:
        cases=delivered_datasets() if a.delivered else datasets(a.archive)
    for case,kind,metric,x,y in cases:
        folder=a.root/case
        folder.mkdir()
        idx=distinct(x,y)
        train,outer=train_test_split(idx,test_size=.25,stratify=y[idx],random_state=261126)
        packet(folder/'train.zip',kind,metric,x[train],y[train])
        packet(folder/'input.zip',kind,metric,x[outer])
        truth=a.root/'evaluation'/(case+'.npy')
        np.save(truth,y[outer],allow_pickle=False)
        inputs[case]=dict(train=len(train),outer=len(outer),metric=metric,train_sha=sha(folder/'train.zip'),
                          input_sha=sha(folder/'input.zip'),truth_sha=sha(truth))
    save(a.root/'protocol.json',dict(source=source(a.delivered),inputs=inputs,smoke=smoke))


def settings(a,work,folder,mode):
    s=env(a.weights,work/'fits.jsonl')
    for key in ('SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX',
                'SUPERAGENT_TASK_BUDGET_ID','AB_B1_PER_WORKER_FIT_LIMIT','SUPERAGENT_RESEARCH_PAIRED_REFLECTION','REFLECTION_STUDY_LIVE'):
        s.pop(key,None)
    s.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT))),
             PROPOSAL_STUDY_OUTER_LABELS=str(a.root/'evaluation'),PROPOSAL_STUDY_OUTER_FEATURES=str(folder/'input.zip'))
    if mode in PAID_MODES:
        s.update(REFLECTION_STUDY_LIVE='1',SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(a.ledger),
                 SUPERAGENT_TASK_BUDGET_ID='proposal-value-'+folder.name,SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file),
                 NO_PROXY='aigc-api.hkust-gz.edu.cn',no_proxy='aigc-api.hkust-gz.edu.cn')
    return s


def run_one(a,case,mode):
    protocol=read(a.root/'protocol.json')
    spec=protocol['inputs'][case]
    folder=a.root/case
    work=folder/mode
    require(protocol['source']==source(a.delivered) and spec['train_sha']==sha(folder/'train.zip') and spec['input_sha']==sha(folder/'input.zip'),
            'Source or task packets changed')
    if (work/'sealed.json').exists():
        return read(work/'sealed.json')
    work.mkdir(exist_ok=True)
    s=settings(a,work,folder,mode)
    run=work/'run'
    delivered,replay=(work/k/'predictions.csv' for k in ('delivery','replay'))
    limit=8 if protocol['smoke'] else 16
    commands=[('solve',delivered,[folder/'train.zip','--prediction-input',folder/'input.zip','--run-dir',run,
                          '--output',delivered,'--fit-limit',limit,'--time-limit',600,'--feedback-reflection',mode]),
              ('predict',replay,[folder/'input.zip','--run-dir',run,'--output',replay])]
    for command,dest,arguments in commands:
        if not dest.exists():
            with (work/(command+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-m','unified_agent',command,*map(str,arguments)],cwd=ROOT,env=s,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=670,check=True)
    agent=run/'agent'
    final,outcome=read(agent/'final_selection.json'),read(agent/'run_outcome.json')
    physical=work/'fits.jsonl'
    fits=len(physical.read_text().splitlines()) if physical.exists() else 0
    require(fits==outcome['attempted_classifier_fits']<=limit and sha(delivered)==sha(replay)
            and sha(agent/final['model_file'])==final['model_sha256'] and not outcome['formal_test_accessed'],'Unverified fitting or delivery')
    if outcome['status']=='completed':
        require(final['refit'] and final['training_rows']==spec['train'] and fits==outcome['completed_classifier_fits'],'Incomplete final refit')
    reserve=read(agent/'proposal_partition.json')
    require(reserve['status']=='sealed','The preregistered panel requires an independent reserve')
    excluded=set(reserve['sealed'])
    for path in agent.glob('*_split.json'):
        split=read(path)
        require(not excluded.intersection(split['fit']+split['held']),'Outer training reserve accessed by search')
    records=read(agent/'experiment_proposal.json')
    history=read(agent/'history.json')
    decisions=read(agent/'decisions.json')
    require(all(d['observed_trials']==[r['trial'] for r in history[:i]] for i,d in enumerate(decisions)),'Future trial used')
    events=[json.loads(s) for s in (agent/'llm_usage.jsonl').read_text().splitlines()]
    calls=[e for e in events if e.get('budget_request_id')]
    require(len(calls)==outcome['new_api_requests']<=int(mode in PAID_MODES),'Unmatched model requests')
    if records['triggered']:
        req=read(agent/'experiment_proposal_input.json')
        require(records['input_digest']==identity(req) and max(req['reference_trials'])<=records['trigger_after_trials'],'Future proposal evidence')
        if mode in DELIVERED_MODES:
            require(records['brief_digest']==identity(read(agent/'proposal_brief.json')),'Changed proposal brief')
    if records['seal_opened']:
        require((agent/'guard_choice_before_labels.json').is_file() and mode!='evidence-control','Ungrounded confirmation')
    receipt=dict(status=outcome['status'],error=outcome.get('error'),model=final['model_sha256'],prediction=sha(delivered),
                 primary_split=sha(agent/'primary_split.json'),reserve=identity(reserve),calls=calls,unknown=outcome['unresolved_api_requests'],
                 fits=fits,wall=outcome.get('wall_seconds'),review=records,
                 core_trace=identity([[d['choice'][k] for k in ('action','members','split')] for d in decisions]),
                 published_seconds=read(agent/'proposal_checkpoints.json')[-1]['elapsed'])
    save(work/'sealed.json',receipt)
    print(case,mode,receipt['status'],fits,'fits',len(calls),'calls',records.get('reply_status'),records.get('guard_action'),flush=True)
    return receipt


def reload_at(a,case,mode,key,snapshot):
    folder=a.root/case
    agent=folder/mode/'run/agent'
    record=snapshot['selection']
    target=a.root/'interim'/case/mode/key
    target.mkdir(parents=True,exist_ok=True)
    name=record['model_file']
    require(sha(agent/name)==record['model_sha256'],'Interim model changed')
    for link in ('task_spec.json',name):
        if not (target/link).exists():
            os.link(agent/link,target/link)
    save(target/'final_selection.json',record)
    result=target/'predictions.csv'
    if not result.exists():
        create_predictions(str(folder/'input.zip'),target,result)
    return dict(path=str(result.relative_to(a.root)),sha=sha(result),rows_trained=record['training_rows'],t=snapshot['elapsed'])


def freeze(a):
    frozen={}
    for case in a.cases:
        arms={}
        for mode in a.modes:
            work=a.root/case/mode
            r=read(work/'sealed.json')
            trace=read(work/'run/agent/proposal_checkpoints.json')
            # Snapshots are recorded on publication, including the initial class prior and full delivery.
            four=next((s for s in trace if s['completed']>=4),trace[-1])
            timed=next((s for s in reversed(trace) if s['elapsed']<=30.),None)
            row=dict(final=dict(path=str((work/'delivery/predictions.csv').relative_to(a.root)),sha=r['prediction']),
                     fourth=reload_at(a,case,mode,'fourth',four),
                     seconds30=reload_at(a,case,mode,'seconds30',timed) if timed is not None else None)
            if a.delivered:
                protected=next((s for s in trace if s['selection']['model_file']=='core_ready_selected_model.joblib'),None)
                if protected is not None:
                    ready=protected['selection']
                    require(ready==read(work/'run/agent/core_ready_final_selection.json') and
                            ready['delivery_status']=='partial' and ready['model_sha256']==r['review']['protected_model'] and
                            ready['training_rows']==len(read(work/'run/agent/proposal_partition.json')['pool']) and
                            protected['elapsed']<=r['review']['decision_seconds'],'Unverified protected delivery')
                    row['protected']=reload_at(a,case,mode,'protected',protected)
            arms[mode]=row
        frozen[case]=arms
    save(a.root/'frozen_predictions.json',frozen)


def score(a):
    frozen=read(a.root/'frozen_predictions.json')
    protocol=read(a.root/'protocol.json')
    report,paired,request_ids={},{},{}
    fits=0
    for case,arms in frozen.items():
        spec=protocol['inputs'][case]
        truth=a.root/'evaluation'/(case+'.npy')
        require(sha(truth)==spec['truth_sha'],'Outer outcomes changed')
        y=np.load(truth,allow_pickle=False)
        measures={}
        rows={}
        receipts=[read(a.root/case/m/'sealed.json') for m in a.modes]
        for key in ('primary_split','core_trace','reserve'):
            require(len({r[key] for r in receipts})==1,'Original route or data differed between arms')
        require(len({r['review'].get('input_digest') for r in receipts})==1,'Unequal proposal inputs')
        if a.delivered:
            require(len({r['review'].get('brief_digest') for r in receipts})==1,'Unequal proposal brief')
        for mode,receipt in zip(a.modes,receipts):
            pred={key:get_predictions(a.root/info['path'],y,info['sha']) for key,info in arms[mode].items() if info}
            measures[mode]=pred
            calls=[]
            for e in receipt['calls']:
                key=e['budget_request_id']
                require(key not in request_ids,'Duplicate billing')
                request_ids[key]=(case,e)
                calls.append({k:e.get(k) for k in ('served_model','input_tokens','output_tokens','course_cost_rmb','latency_seconds','finish_reason')})
            gains={k:float(score_from_counts(y,p,spec['metric'])) for k,p in pred.items()}
            rows[mode]=dict(scores=gains,fits=receipt['fits'],status=receipt['status'],error=receipt['error'],usage=calls,
                            wall=receipt['wall'],model_published_seconds=receipt['published_seconds'],review=receipt['review'])
            fits+=receipt['fits']
        comparisons={}
        contrasts=[('Flash-local',DELIVERED_MODES[1],DELIVERED_MODES[0])] if a.delivered else [
            ('B-A',MODES[1],MODES[0]),('C-B',MODES[2],MODES[1]),('C-A',MODES[2],MODES[0])]
        for key,first,second in contrasts:
            comp={}
            for cut in measures[first].keys() & measures[second].keys():
                new,old=measures[first][cut],measures[second][cut]
                delta=rows[first]['scores'][cut]-rows[second]['scores'][cut]
                paired.setdefault(key,{}).setdefault(cut,[]).append(delta)
                comp[cut]=dict(delta=delta,changed=int(sum(new!=old)),corrected=int(sum((new==y)&(old!=y))),spoiled=int(sum((new!=y)&(old==y))))
            comparisons[key]=comp
        report[case]=dict(metric=spec['metric'],outer=len(y),train=spec['train'],arms=rows,comparisons=comparisons)
    with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        req=[dict(row) for row in db.execute('SELECT * FROM requests') if row['scope'] in {'proposal-value-'+c for c in a.cases}]
        require({r['id'] for r in req}==request_ids.keys(),'A request is missing from the batch')
        transitions=[dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY seq') if r['request_id'] in request_ids]
    inputs=outputs=unknown=0
    for r in req:
        case,e=request_ids[r['id']]
        role='protected_experiment_proposal' if a.delivered else 'experiment_proposal'
        require(r['scope']=='proposal-value-'+case and e['role']==role,'Unexpected model scope or role')
        if r['status']=='known':
            require(e['served_model'].casefold()=='deepseek-v4.1-flash' and r['course_units']==1000*(e['input_tokens']+4*e['output_tokens']),
                    'Model identity or course token accounting mismatch')
            settled=json.loads(r['payload'])['resolution']
            require(all(settled[k]==e[k] for k in ('input_tokens','output_tokens')),'Settlement token counts differ')
            inputs+=e['input_tokens']
            outputs+=e['output_tokens']
        else:
            unknown+=1
    save(a.root/'pending_merge.json',dict(team_merge_state='pending_team_handoff',requests=req,transitions=transitions))
    require(fits==sum(len(p.read_text().splitlines()) for p in a.root.glob('*/*/fits.jsonl'))<=16*len(a.cases)*len(a.modes)
            and len(req)<=len(a.cases),'Batch ceiling')
    stats={k:{cut:dict(mean=float(np.mean(values)),worst=min(values),tasks=len(values)) for cut,values in by.items()} for k,by in paired.items()}
    reviews=[v['arms'][a.modes[-1]]['review'] for v in report.values()]
    tested=sum(bool(r.get('proposal_used') and r.get('screen')) for r in reviews)
    distinct_tested=sum(bool(r.get('proposal_used') and r.get('screen') and r['tested_candidate']!=r['local_alternative']) for r in reviews)
    complete=all(r['status']=='completed' for v in report.values() for r in v['arms'].values())
    if a.delivered:
        comparison=stats['Flash-local']['final']
        engineering=bool(len(req)==len(a.cases) and not unknown and complete and
                         all(r.get('decision_time_status')=='valid' for r in reviews) and
                         all(v['comparisons']['Flash-local'].get('protected',{}).get('changed')==0 for v in report.values()))
        success=engineering and distinct_tested>0 and comparison['mean']>0 and comparison['worst']>=-.01
    else:
        success=bool(not unknown and distinct_tested and complete and all(stats[k]['final']['mean']>0 and
                     min(stats[k]['final']['worst'],stats[k]['fourth']['worst'])>=-.01 for k in ('C-B','C-A')))
    result=dict(protocol=sha(DELIVERED_PLAN if a.delivered else PLAN),source=protocol['source'],cases=report,comparisons=stats,actual_fits=fits,requests=len(req),
                input_tokens=inputs,output_tokens=outputs,known_course_rmb=(inputs+4*outputs)/1e6,unknown=unknown,
                executed_proposals=tested,distinct_executed_proposals=distinct_tested,meets_criterion=success,formal_test_accessed=False)
    if a.delivered:
        result['engineering_criterion']=engineering
    save(a.root/'completion.json',result)
    save(a.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','smoke','local','paid','freeze','score'))
    p.add_argument('root',type=lambda x:Path(x).resolve())
    p.add_argument('--delivered',action='store_true',help='Run the separately registered protected-delivery follow-up')
    for key in ('weights','archive','ledger','key-file','output'):
        p.add_argument('--'+key,type=lambda x:Path(x).resolve())
    a=p.parse_args()
    a.cases,a.modes=(DELIVERED_CASES,DELIVERED_MODES) if a.delivered else (CASES,MODES)
    if a.stage=='prepare':
        prepare(a)
    elif a.stage=='smoke':
        prepare(a,smoke=True)
        require(run_one(a,'smoke','evidence-local')['status']=='completed','Anonymous CLI smoke failed')
    elif a.stage=='local':
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda pair:run_one(a,*pair),[(c,m) for c in a.cases for m in a.modes[:-1]]))
    elif a.stage=='paid':
        for case in a.cases:
            with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
                require(not db.execute("SELECT 1 FROM requests WHERE status!='known'").fetchone(),'New paid dispatch halted on unresolved usage')
            require(not run_one(a,case,a.modes[-1])['unknown'],'Unresolved request; stop paid dispatch')
    elif a.stage=='score':
        score(a)
    elif os.environ.get('AB_B1_OFFLINE')=='1':
        freeze(a)
    else:
        fitlog=a.root/'unexpected_checkpoint_fits.jsonl'
        subprocess.run([sys.executable,'-m','scripts.run_proposal_value_study',*sys.argv[1:]],cwd=ROOT,
                       env=env(a.weights,fitlog),check=True)
        require(not fitlog.exists() or not fitlog.read_text().strip(),'Unexpected fit during replay')
