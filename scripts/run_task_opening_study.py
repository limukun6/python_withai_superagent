"""Cold runs of the opening reader on real anonymous tasks, with frozen external predictions and separate billing."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from scripts.run_conditional_review_study import get_predictions
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, env, make_zip, read, require, save, sha
from scripts.verify_conditional_study_receipts import score_from_counts
from superagent.experiment_policy import identity


PLAN=ROOT/'research/task_opening_1126/protocol.json'
CASES=('s91','s92','s93','s94')
MODES=('local','flash')
ARCHIVES={'recordings':'f440702d581b8d7edf2cd7ed336abe3211fc686792eef3b2665a6f3008bbbc81',
          'comments':'bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a',
          'digits':'6fb091394714cddda5751d4e1c2781ab094e7cf15de07917fb40e581f19efc75',
          'signals':'afb3e4f3674409642e6606e14faf4cb66edc3a8c74034839f1cee4e017c59a4e'}


def subset(ids,y,limit,seed):
    if len(ids)<=limit:
        return ids
    return train_test_split(ids,train_size=limit,stratify=y[ids],random_state=seed)[0]


def sources():
    audit=[Path(__file__),ROOT/'scripts/reflection_study_observer/sitecustomize.py',ROOT/'scripts/ab_acceptance_guard/sitecustomize.py',
           ROOT/'scripts/run_planner_self_review.py',ROOT/'scripts/unified_released_study.py',PLAN]
    return dict(agent=source_identity(ROOT),study=identity({str(p.relative_to(ROOT)):sha(p) for p in audit}))


def datasets(directory):
    for name,digest in ARCHIVES.items():
        require(sha(directory/(name+'.zip'))==digest,'Unexpected public archive: '+name)
    with zipfile.ZipFile(directory/'recordings.zip') as z:
        rows=list(csv.DictReader(io.TextIOWrapper(z.open('Frogs_MFCCs.csv'),encoding='utf-8-sig')))
    features=[k for k in rows[0] if k.startswith('MFCCs')]
    x=np.asarray([[float(r[k]) for k in features] for r in rows])
    y=np.asarray([int(r['Family']=='Leptodactylidae') for r in rows])
    groups=np.asarray([r['RecordID'] for r in rows])
    ix=subset(distinct(x,y),y,1600,261191)
    codes={g:int(hashlib.sha256(('opaque-group:'+g).encode()).hexdigest()[:12],16)/2**48 for g in set(groups)}
    x=np.column_stack([x,[codes[g] for g in groups]])
    folds=StratifiedGroupKFold(4,shuffle=True,random_state=261126)
    a,b=next((ix[a],ix[b]) for a,b in folds.split(x[ix],y[ix],groups[ix]) if len(set(y[ix[a]]))==len(set(y[ix[b]]))==2)
    require(not set(groups[a])&set(groups[b]),'Recording crosses the external boundary')
    fields=[dict(name='measure_'+str(i),type='numeric') for i in range(len(features))]+[dict(name='record_code',type='numeric')]
    yield 's91','tabular','balanced_accuracy',x,y,a,b,dict(columns=fields,evaluation_protocol=dict(kind='entity',column='record_code')),(
        'Predict whether an acoustic syllable belongs to the target family. Several syllables can come from one recording. '
        'Evaluate on previously unseen values of record_code. Report balanced accuracy.')
    contents,classes,blocks=[],[],[]
    with zipfile.ZipFile(directory/'comments.zip') as z:
        files=sorted(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX'))
        for i,f in enumerate(files):
            for row in csv.DictReader(io.TextIOWrapper(z.open(f),encoding='utf-8-sig')):
                contents.append(row['CONTENT']); classes.append(int(row['CLASS'])); blocks.append(i)
    x,y,groups=np.asarray(contents),np.asarray(classes),np.asarray(blocks)
    ix=distinct(x,y)
    a=subset(ix[groups[ix]!=len(files)-1],y,760,261192)
    b=subset(ix[groups[ix]==len(files)-1],y,240,261192)
    yield 's92','text','macro_f1',x,y,a,b,{},(
        'Classify whether a public comment is unsolicited promotion. Each comment is written about a video. '
        'New comments can refer to different topics. Report macro F1.')
    with zipfile.ZipFile(directory/'digits.zip') as z:
        data=np.loadtxt(io.BytesIO(z.read('semeion.data')))
    require(data.shape[1]==266 and np.all(np.sum(data[:,-10:],axis=1)==1),'Unexpected digit encoding')
    x,y=(255*data[:,:256]).reshape(-1,16,16).astype('uint8'),data[:,-10:].argmax(axis=1)
    ix=subset(distinct(x,y),y,1200,261193)
    a,b=train_test_split(ix,test_size=.25,stratify=y[ix],random_state=261126)
    yield 's93','image','accuracy',x,y,a,b,{},(
        'Recognize a handwritten numeral from one binary image. Samples can differ in handwriting style and stroke thickness. '
        'Report classification accuracy.')
    with zipfile.ZipFile(directory/'signals.zip') as z:
        data=np.loadtxt(io.BytesIO(z.read('wifi_localization.txt')))
    x,y=data[:,:7],data[:,7].astype(int)
    ix=subset(distinct(x,y),y,1200,261194)
    a,b=train_test_split(ix,test_size=.25,stratify=y[ix],random_state=261126)
    yield 's94','tabular','macro_f1',x,y,a,b,{},(
        'Predict the room from seven received signal strength measurements. Each row is one reading. Report macro F1.')


def write_packet(path,kind,metric,x,y,extras,prose):
    temp=path.parent/(path.stem+'_base.zip')
    packet(temp,kind,metric,x,y)
    try:
        with zipfile.ZipFile(temp) as z:
            manifest=json.loads(z.read('manifest.json'))
            manifest.update(extras)
            members=[(n,z.read(n)) for n in z.namelist() if n!='manifest.json']
        make_zip(path,[('manifest.json',json.dumps(manifest).encode()),('README.md',prose.encode()),*members])
    finally:
        temp.unlink(missing_ok=True)


def prepare(args,smoke=False):
    root=args.root
    root.mkdir(parents=True,exist_ok=False)
    (root/'evaluation').mkdir()
    if smoke:
        rng=np.random.default_rng(1331)
        x=rng.normal(size=(160,7))
        y=(x[:,0]+.6*x[:,1]+rng.normal(size=len(x))>0).astype(int)
        a,b=train_test_split(np.arange(len(x)),stratify=y,test_size=.25,random_state=261126)
        data=[('smoke','tabular','accuracy',x,y,a,b,{},'Predict the binary outcome of a measurement. Report accuracy.')]
    else:
        data=datasets(args.archives)
    specs={}
    for case,kind,metric,x,y,a,b,extra,prose in data:
        require(not set(a)&set(b),'Overlapping row ids')
        dest=root/case
        dest.mkdir()
        write_packet(dest/'train.zip',kind,metric,x[a],y[a],extra,prose)
        write_packet(dest/'input.zip',kind,metric,x[b],None,extra,prose)
        truth=root/'evaluation'/(case+'.npy')
        np.save(truth,y[b],allow_pickle=False)
        specs[case]=dict(metric=metric,train=len(a),outer=len(b),train_sha=sha(dest/'train.zip'),input_sha=sha(dest/'input.zip'),
                         truth_sha=sha(truth),partition=identity([a.tolist(),b.tolist()]))
    save(root/'protocol.json',dict(preregistered=sha(PLAN),source=sources(),inputs=specs,smoke=smoke))


def environment(args,work,folder,paid):
    settings=env(args.weights,work/'fits.jsonl')
    for k in ('SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX',
              'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_PAIRED_REFLECTION','REFLECTION_STUDY_LIVE','AB_B1_PER_WORKER_FIT_LIMIT'):
        settings.pop(k,None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT))),
                    PROPOSAL_STUDY_OUTER_LABELS=str(args.root/'evaluation'),PROPOSAL_STUDY_OUTER_FEATURES=str(folder/'input.zip'))
    if paid:
        settings.update(REFLECTION_STUDY_LIVE='1',SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(args.ledger),
            SUPERAGENT_TASK_BUDGET_ID='task-reading-v1-'+folder.name,SUPERAGENT_PLANNER_KEY_FILE=str(args.key_file),
            NO_PROXY='aigc-api.hkust-gz.edu.cn',no_proxy='aigc-api.hkust-gz.edu.cn')
    return settings


def invoke(settings,log,command):
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','unified_agent',*map(str,command)],cwd=ROOT,env=settings,
                       stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=670)


def run_one(args,case,mode,smoke=False):
    folder=args.root/case
    protocol=read(args.root/'protocol.json')
    spec=protocol['inputs'][case]
    require(sources()==protocol['source'] and sha(folder/'train.zip')==spec['train_sha'] and
            sha(folder/'input.zip')==spec['input_sha'],'Source or input changed')
    work=folder/mode
    if (work/'sealed.json').is_file():
        return read(work/'sealed.json')
    work.mkdir(exist_ok=True)
    run=work/'run'
    agent=run/'agent'
    paid=mode=='flash' and not smoke
    if paid:
        with sqlite3.connect(args.ledger.as_uri()+'?mode=ro',uri=True) as db:
            require(not db.execute("SELECT 1 FROM requests WHERE status!='known'").fetchone(),'Paid queue paused for unresolved usage')
    limit=8 if smoke else 16
    settings=environment(args,work,folder,paid)
    output=work/'delivery/predictions.csv'
    if not output.exists():
        invoke(settings,work/'solve.log',['solve',folder/'train.zip','--run-dir',run,'--prediction-input',folder/'input.zip',
               '--output',output,'--task-analysis',mode,'--fit-limit',limit,'--time-limit',600])
    outcome=read(agent/'run_outcome.json')
    final=read(agent/'final_selection.json')
    opening=read(agent/'task_opening.json') if (agent/'task_opening.json').exists() else {}
    core=read(agent/'opening_incumbent.json') if (agent/'opening_incumbent.json').exists() else None
    delivery_sha=sha(output)
    offline=environment(args,work,folder,False)
    replay=work/'replay/predictions.csv'
    if not replay.exists():
        invoke(offline,work/'replay.log',['predict',folder/'input.zip','--run-dir',run,'--output',replay])
    require(sha(replay)==delivery_sha and sha(agent/final['model_file'])==final['model_sha256'],'Reload or model identity mismatch')
    base_sha=None
    if core:
        local=work/'original/agent'
        local.mkdir(parents=True,exist_ok=True)
        require(core['refit'] and core['training_rows']==spec['train'] and sha(agent/core['model_file'])==core['model_sha256']
                and core['model_sha256']==opening['incumbent_sha'],'Ordinary full-data delivery was lost')
        for name in (core['model_file'],'task_spec.json'):
            if not (local/name).exists():
                os.link(agent/name,local/name)
        save(local/'final_selection.json',core)
        base=work/'original-predictions/predictions.csv'
        if not base.exists():
            invoke(offline,work/'original.log',['predict',folder/'input.zip','--run-dir',local.parent,'--output',base])
        base_sha=sha(base)
    log=work/'fits.jsonl'
    count=len(log.read_text().splitlines()) if log.is_file() else 0
    events=[json.loads(s) for s in (agent/'llm_usage.jsonl').read_text().splitlines()]
    calls=[e for e in events if e.get('budget_request_id')]
    require(count==outcome['attempted_classifier_fits']<=limit and outcome['formal_test_accessed'] is False,'Invalid physical fits/access')
    require(len(calls)==outcome['new_api_requests']<=int(paid),'Unexpected billable request')
    if outcome['status']=='completed':
        require(core and final['refit'] and final['training_rows']==spec['train'] and count==outcome['completed_classifier_fits'],'Incomplete training')
    decisions=read(agent/'decisions.json') if (agent/'decisions.json').exists() else []
    brief=read(agent/'opening_brief.json') if (agent/'opening_brief.json').exists() else {}
    if opening:
        require(opening['brief_digest']==identity(brief),'Unmatched task brief')
    receipt=dict(status=outcome['status'],error=outcome.get('error'),final=delivery_sha,base=base_sha,fits=count,
                 wall=outcome.get('wall_seconds'),calls=calls,unknown=outcome['unresolved_api_requests'],opening=opening,
                 decisions=identity([[d['choice'][k] for k in ('action','members','split')] for d in decisions]),
                 folds=identity({p.name:sha(p) for p in agent.glob('*_split.json')}))
    save(work/'sealed.json',receipt)
    print(case,mode,outcome['status'],count,'fits',len(calls),'calls',opening.get('check_status'),opening.get('result'),flush=True)
    return receipt


def freeze(args):
    manifest={}
    for case in CASES:
        entries={m:read(args.root/case/m/'sealed.json') for m in MODES}
        if all(r['status']=='completed' for r in entries.values()):
            for key in ('base','decisions','folds'):
                require(entries['local'][key]==entries['flash'][key],'Core replay differed: '+key)
            require(entries['local']['opening']['brief_digest']==entries['flash']['opening']['brief_digest'],'Unequal opening evidence')
        manifest[case]={m:dict(sealed_sha=sha(args.root/case/m/'sealed.json'),final=r['final'],original=r['base']) for m,r in entries.items()}
    save(args.root/'all_predictions_frozen.json',manifest)


def score(args):
    frozen=read(args.root/'all_predictions_frozen.json')
    specs=read(args.root/'protocol.json')['inputs']
    results={}
    submitted={}
    for case in CASES:
        truth=args.root/'evaluation'/(case+'.npy')
        require(sha(truth)==specs[case]['truth_sha'],'Changed development truth')
        y=np.load(truth,allow_pickle=False)
        modes={}
        paired={}
        metric=specs[case]['metric']
        function={'accuracy':accuracy_score,'balanced_accuracy':balanced_accuracy_score,'macro_f1':lambda y,p:f1_score(y,p,average='macro')}[metric]
        for mode in MODES:
            work=args.root/case/mode
            seal=frozen[case][mode]
            require(sha(work/'sealed.json')==seal['sealed_sha'],'Changed run receipt')
            r=read(work/'sealed.json')
            row=dict(status=r['status'],fits=r['fits'],wall=r['wall'],error=r['error'],opening=r['opening'])
            for key,directory in [('final','delivery'),('original','original-predictions')]:
                if seal[key] is not None:
                    p=get_predictions(work/directory/'predictions.csv',y,seal[key])
                    val=float(score_from_counts(y,p,metric))
                    require(abs(val-function(y,p))<1e-12,'Independent metric disagrees')
                    row[key]=val
                    if key=='final':
                        paired[mode]=p
            usage=[]
            for event in r['calls']:
                req=event['budget_request_id']
                require(req not in submitted and event['role']=='task_opening' and mode=='flash','Unexpected call')
                submitted[req]=(case,event)
                usage.append({k:event.get(k) for k in ('served_model','input_tokens','output_tokens','latency_seconds','finish_reason','course_cost_rmb')})
            row['usage']=usage
            modes[mode]=row
        local,flash=paired['local'],paired['flash']
        results[case]=dict(metric=metric,train=specs[case]['train'],outer=len(y),arms=modes,
            delta=modes['flash']['final']-modes['local']['final'],changed=int(sum(flash!=local)),
            corrected=int(sum((flash==y)&(local!=y))),spoiled=int(sum((flash!=y)&(local==y))))
    with sqlite3.connect(args.ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        records=[dict(r) for r in db.execute('SELECT * FROM requests') if r['scope'] in {'task-reading-v1-'+s for s in CASES}]
        transitions=[dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY seq') if r['request_id'] in submitted]
    require({r['id'] for r in records}==set(submitted),'Missing or duplicate billing')
    tokens_in=tokens_out=unknown=0
    for row in records:
        case,e=submitted[row['id']]
        require(row['scope']=='task-reading-v1-'+case,'Wrong billing scope')
        if row['status']=='known':
            settled=json.loads(row['payload'])['resolution']
            require(e['served_model'].casefold()=='deepseek-v4.1-flash' and
                    all(e[k]==settled[k] for k in ('input_tokens','output_tokens')) and
                    row['course_units']==1000*(e['input_tokens']+4*e['output_tokens']),'Served model or token charge mismatch')
            tokens_in+=e['input_tokens'];tokens_out+=e['output_tokens']
        else:
            unknown+=1
    save(args.root/'pending_merge.json',dict(requests=records,transitions=transitions,state='pending_team_handoff'))
    changes=[v['delta'] for v in results.values()]
    distinct=sum(bool(r['arms']['flash']['opening'].get('changed_experiment') and r['arms']['flash']['opening'].get('comparisons')) for r in results.values())
    complete=all(v['status']=='completed' for r in results.values() for v in r['arms'].values())
    output=dict(protocol=sha(PLAN),source=read(args.root/'protocol.json')['source'],cases=results,complete=complete,
                actual_fits=sum(v['fits'] for r in results.values() for v in r['arms'].values()),
                calls=len(records),input_tokens=tokens_in,output_tokens=tokens_out,known_course_rmb=(tokens_in+4*tokens_out)/1e6,
                unknown=unknown,changed_experiments=distinct,mean_delta=float(np.mean(changes)),worst_delta=min(changes),
                meets_quality_criterion=bool(complete and distinct and not unknown and np.mean(changes)>0 and min(changes)>=-.01),
                formal_test_accessed=False)
    save(args.root/'completion.json',output)
    save(args.output,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','smoke','local','flash','freeze','score'))
    parser.add_argument('root',type=lambda p:Path(p).resolve())
    for arg in ('archives','weights','ledger','key-file','output'):
        parser.add_argument('--'+arg,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    if args.stage=='smoke':
        prepare(args,True)
        require(run_one(args,'smoke','flash',True)['status']=='completed','Offline full-entry smoke failed')
    elif args.stage=='prepare':
        prepare(args)
    elif args.stage in MODES:
        for case in CASES:
            r=run_one(args,case,args.stage)
            if args.stage=='flash':
                require(r['unknown']==0,'New unresolved usage; stopped paid dispatch')
    elif args.stage=='freeze':
        freeze(args)
    else:
        score(args)
