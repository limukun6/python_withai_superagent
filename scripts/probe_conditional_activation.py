"""Prospectively probe natural activation without using evaluation labels or selecting a successful response."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from scripts.run_conditional_review_study import run_one, source
from scripts.run_planner_self_review import packet
from scripts.unified_released_study import ROOT, read, require, save, sha


PLAN=ROOT/'research/conditional_review_1125/activation_protocol.json'
IDS=('n71','n72','n73')


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False)
    save(a.root/'prior.json',read(a.main/'prior.json'))
    cases={}
    for case,seed in zip(IDS,(26171,26172,26173)):
        folder=a.root/case
        folder.mkdir()
        rng=np.random.default_rng(seed)
        x=rng.normal(size=(300,9))
        signal=np.where(x[:,3]>0,.8*x[:,0]-.5*x[:,1],x[:,0]*x[:,1]+.45*np.sin(2*x[:,2]))
        y=((signal>0) ^ (rng.random(len(x))<.24)).astype(int)
        query=rng.normal(size=(80,9))
        packet(folder/'train.zip','tabular','balanced_accuracy',x,y)
        packet(folder/'input.zip','tabular','balanced_accuracy',query)
        cases[case]=dict(metric='balanced_accuracy',train=len(x),outer=len(query),
                         train_sha=sha(folder/'train.zip'),input_sha=sha(folder/'input.zip'))
    save(a.root/'protocol.json',dict(source=source(),prior=sha(a.root/'prior.json'),inputs=cases,
                                     extension_protocol=sha(PLAN),extension_runner=sha(Path(__file__))))


def run(a):
    receipts=[read(a.main/c/'conditional-flash/sealed.json') for c in read(a.main/'protocol.json')['inputs']]
    require(not any(r['calls'] or r['unknown'] or r['review']['status']!='skipped' for r in receipts),
            'The main panel already exercised a conditional request or remains incomplete')
    protocol=read(a.root/'protocol.json')
    require(protocol['extension_protocol']==sha(PLAN) and protocol['extension_runner']==sha(Path(__file__)),'Probe contract changed')
    outcomes={}
    with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
        require(not db.execute("SELECT 1 FROM requests WHERE status!='known'").fetchone(),'Unresolved local usage')
    for case in IDS:
        r=run_one(a,case,'conditional-flash')
        usage=[]
        for e in r['calls']:
            with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
                scope,status,units,payload=db.execute('SELECT scope,status,course_units,payload FROM requests WHERE id=?',
                                                     (e['budget_request_id'],)).fetchone()
            require(scope==r['scope'] and json.loads(payload).get('paired_reflection_arm')=='conditional-flash','Incorrect task scope')
            if status=='known':
                require(e['role']=='feedback_reflection' and e['served_model'].casefold()=='deepseek-v4.1-flash'
                        and units==1000*(e['input_tokens']+4*e['output_tokens']), 'Course accounting does not match')
            usage.append({k:e.get(k) for k in ('served_model','input_tokens','output_tokens','course_cost_rmb')})
        outcomes[case]=dict(review=r['review'],fits=r['fits'],usage=usage,unknown=r['unknown'],prediction=r['prediction'])
        if r['calls'] or r['unknown'] or r['review']['status']!='skipped':
            break
    fits=sum(r['fits'] for r in outcomes.values())
    calls=sum(len(r['usage']) for r in outcomes.values())
    require(calls<=1 and fits==sum(len(p.read_text().splitlines()) for p in a.root.glob('*/*/fits.jsonl'))<=48,'Study bounds exceeded')
    result=dict(outcomes=outcomes,fits=fits,calls=calls,held_out_labels_created=False,plan=sha(PLAN))
    save(a.root/'completion.json',result)
    save(a.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','run'))
    p.add_argument('root',type=lambda x:Path(x).resolve())
    for name in ('main','weights','ledger','key-file','output'):
        p.add_argument('--'+name,type=lambda x:Path(x).resolve())
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a)
    else:
        run(a)
