"""Compare interim publication on already-planned measurements with unaltered exploration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

from scripts.audit_interrupted_reflection import freeze as recover_primary
from scripts.run_conditional_review_study import get_predictions, run_one, source
from scripts.run_planner_self_review import packet
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_feedback_reflection_study import independent_score
from superagent.predict import create_predictions


PLAN=ROOT/'research/conditional_review_1125/scheduled_protocol.json'
FRESH=('n74','n75')
CASES=('n71',*FRESH)
MODE='scheduled-flash'


def draws(rng,n):
    x=rng.normal(size=(n,9))
    signal=np.where(x[:,3]>0,.8*x[:,0]-.5*x[:,1],x[:,0]*x[:,1]+.45*np.sin(2*x[:,2]))
    return x,((signal>0) ^ (rng.random(n)<.24)).astype(int)


def prepare(a):
    root=a.root
    root.mkdir(parents=True,exist_ok=False)
    (root/'evaluation').mkdir()
    shutil.copyfile(a.previous/'prior.json',root/'prior.json')
    inputs={}
    for case in CASES:
        folder=root/case
        folder.mkdir()
        if case=='n71':
            for name in ('train.zip','input.zip'):
                shutil.copyfile(a.previous/case/name,folder/name)
            inputs[case]=read(a.previous/'protocol.json')['inputs'][case]
        else:
            rng=np.random.default_rng(26100+int(case[1:]))
            x,y=draws(rng,300)
            query,truth=draws(rng,160)
            packet(folder/'train.zip','tabular','balanced_accuracy',x,y)
            packet(folder/'input.zip','tabular','balanced_accuracy',query)
            path=root/'evaluation'/(case+'.npy')
            np.save(path,truth,allow_pickle=False)
            inputs[case]=dict(metric='balanced_accuracy',train=len(y),outer=len(truth),truth_sha=sha(path),
                              train_sha=sha(folder/'train.zip'),input_sha=sha(folder/'input.zip'))
    save(root/'protocol.json',dict(source=source(),prior=sha(root/'prior.json'),inputs=inputs,
                                   study=sha(PLAN),runner=sha(Path(__file__))))


def paired(a):
    for case in CASES:
        with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
            require(not db.execute("SELECT 1 FROM requests WHERE status!='known'").fetchone(),'Unresolved local usage')
        receipt=run_one(a,case,MODE)
        require(not receipt['unknown'],'Unresolved new request')


def reload_pointer(folder,agent,pointer,query):
    record=read(agent/pointer)
    name=record['model_file']
    require(sha(agent/name)==record['model_sha256'],'Changed interim model')
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'final_selection.json').exists():
        os.link(agent/'task_spec.json',folder/'task_spec.json')
        os.link(agent/name,folder/name)
        save(folder/'final_selection.json',record)
    require(read(folder/'final_selection.json')==record,'Interim pointer changed')
    result=folder/'predictions.csv'
    if not result.exists():
        create_predictions(str(query),folder,result)
    require(read(folder/'prediction_manifest.json')['delivery_status']=='partial','Wrong pointer type')
    return result


def freeze(a):
    root=a.root
    frozen={}
    for case in CASES:
        folder=root/case
        control=(a.previous/case/'conditional-flash' if case=='n71' else folder/'control')
        standard=read(control/'sealed.json')
        live=read(folder/MODE/'sealed.json')
        require(standard['split']==live['split'] and standard['first_three']==live['first_three'],'Different paired data/start')
        item=dict(full_search_equal=standard['pre_review_search']==live['pre_review_search'],
                  control_final=standard['prediction'],review_final=live['prediction'],
                  control_fits=standard['fits'],review_fits=live['fits'])
        require(sha(control/'delivery/predictions.csv')==item['control_final'],'Changed control predictions')
        require(sha(folder/MODE/'delivery/predictions.csv')==item['review_final'],'Changed reviewed predictions')
        trigger=live['review'].get('trial')
        if trigger is not None:
            snapshots={}
            agent=folder/MODE/'run/agent'
            for key in ('before','after'):
                path=reload_pointer(root/'interim'/case/key,agent,'scheduled_selection_'+key+'.json',folder/'input.zip')
                snapshots[key]=dict(file=str(path.relative_to(root)),digest=sha(path))
            # Rebuild the ordinary published checkpoint from control history at the exact matched trial.
            recover=root/'control_recovery'/case
            recover.mkdir(parents=True,exist_ok=True)
            (recover/case).mkdir(exist_ok=True)
            for name,target in [('train.zip',folder/'train.zip'),('input.zip',folder/'input.zip'),('control',control)]:
                link=recover/case/name
                if not link.exists():
                    link.symlink_to(target)
            save(recover/'all_predictions_frozen.json',{case:{'control':standard['prediction']}})
            out=recover/'partial'
            out.mkdir(exist_ok=True)
            recover_primary(recover,out,(trigger,))
            rec=read(out/'frozen.json')[case]['control'][str(trigger)]
            snapshots['control']=dict(file=str((out/rec['file']).relative_to(root)),digest=rec['digest'])
            require(snapshots['before']['digest']==snapshots['control']['digest'],'Pre-review delivery differs from matched control')
            item['interim']=snapshots
        frozen[case]=item
    save(root/'frozen.json',frozen)


def score(a):
    root=a.root
    protocol=read(root/'protocol.json')
    frozen=read(root/'frozen.json')
    records={}
    tokens_in=tokens_out=calls=fits=0
    latency=0.
    for case in CASES:
        folder=root/case
        item=frozen[case]
        receipt=read(folder/MODE/'sealed.json')
        bill=[]
        with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
            rows=db.execute('SELECT id,status,course_units FROM requests WHERE scope=?',(receipt['scope'],)).fetchall()
        require({r[0] for r in rows}=={e['budget_request_id'] for e in receipt['calls']},'Missing or repeated scope usage')
        for e in receipt['calls']:
            units=next(u for key,s,u in rows if key==e['budget_request_id'] and s=='known')
            require(e['role']=='feedback_reflection' and e['served_model'].casefold()=='deepseek-v4.1-flash'
                    and units==1000*(e['input_tokens']+4*e['output_tokens']),'Unverified campus price or returned model')
            bill.append({k:e.get(k) for k in ('served_model','input_tokens','output_tokens','course_cost_rmb','latency_seconds','finish_reason')})
            tokens_in+=e['input_tokens']
            tokens_out+=e['output_tokens']
            latency+=e['latency_seconds']
        calls+=len(bill)
        fits+=item['review_fits']+(item['control_fits'] if case in FRESH else 0)
        record=dict(replay=case not in FRESH,trigger_trial=receipt['review'].get('trial'),review=receipt['review'],
                    usage=bill,fit_counts=[item['control_fits'],item['review_fits']],search_equal=item['full_search_equal'],
                    final_predictions_equal=item['control_final']==item['review_final'])
        if 'interim' in item:
            interim=item['interim']
            proxy=np.zeros(protocol['inputs'][case]['outer'],dtype=int)
            preds={k:get_predictions(root/r['file'],proxy,r['digest']) for k,r in interim.items()}
            record['interim_changed_rows']=int(np.sum(preds['control']!=preds['after']))
        if case in FRESH:
            truth=root/'evaluation'/(case+'.npy')
            require(sha(truth)==protocol['inputs'][case]['truth_sha'],'Held-out labels changed')
            y=np.load(truth,allow_pickle=False)
            pc=get_predictions(folder/'control/delivery/predictions.csv',y,item['control_final'])
            pr=get_predictions(folder/MODE/'delivery/predictions.csv',y,item['review_final'])
            record.update(outer_rows=len(y),final_scores=[independent_score('balanced_accuracy',y,p) for p in (pc,pr)],
                          final_changed_rows=int(sum(pc!=pr)))
            if 'interim' in item:
                record['interim_scores']=[independent_score('balanced_accuracy',y,preds[k]) for k in ('control','after')]
                record['interim_corrected']=int(sum((preds['after']==y)&(preds['control']!=y)))
                record['interim_spoiled']=int(sum((preds['after']!=y)&(preds['control']==y)))
        records[case]=record
    require(fits==sum(len(p.read_text().splitlines()) for p in root.glob('*/*/fits.jsonl'))<=80 and calls<=3,'Unexpected costs')
    interim=[records[c]['interim_scores'] for c in FRESH if 'interim_scores' in records[c]]
    passes=any(new>old for old,new in interim) and all(new>=old for old,new in interim) and all(
        records[c]['final_scores'][1]>=records[c]['final_scores'][0] for c in FRESH)
    result=dict(records=records,fits=fits,calls=calls,input_tokens=tokens_in,output_tokens=tokens_out,
                latency_seconds=round(latency,3),known_course_rmb=(tokens_in+4*tokens_out)/1e6,
                unresolved=0,formal_test_accessed=False,fresh_signal=passes,protocol_sha256=sha(PLAN))
    save(root/'completion.json',result)
    save(a.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','control','paid','freeze','score'))
    p.add_argument('root',type=lambda x:Path(x).resolve())
    for arg in ('previous','weights','ledger','key-file','output'):
        p.add_argument('--'+arg,type=lambda x:Path(x).resolve())
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a)
    else:
        contract=read(a.root/'protocol.json')
        require(contract['study']==sha(PLAN) and contract['runner']==sha(Path(__file__)) and contract['source']==source(),'Frozen study changed')
        if a.stage=='control':
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda case:run_one(a,case,'control'),FRESH))
        elif a.stage=='paid':
            paired(a)
        elif a.stage=='score':
            score(a)
        elif os.getenv('AB_B1_OFFLINE')=='1':
            freeze(a)
        else:
            fitlog=a.root/'unexpected_posthoc_fits.jsonl'
            subprocess.run([sys.executable,'-m','scripts.run_scheduled_review_study',*sys.argv[1:]],cwd=ROOT,check=True,
                           env=env(a.weights,fitlog))
            require(not fitlog.exists() or not fitlog.read_text().strip(),'Unexpected training during checkpoint verification')
