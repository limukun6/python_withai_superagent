"""Independently check the scores and one-scope accounting; keep the team's merge rows local."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from scripts.unified_released_study import read, require, save, sha


def score_from_counts(y,p,metric):
    labels=np.unique(y)
    require(set(np.unique(p))<=set(labels),'Unknown predicted class')
    cm=np.array([[int(sum((y==a)&(p==b))) for b in labels] for a in labels])
    if metric=='accuracy':
        return cm.trace()/cm.sum()
    if metric=='balanced_accuracy':
        return np.mean(cm.diagonal()/cm.sum(axis=1))
    if metric=='macro_f1':
        return np.mean(2*cm.diagonal()/(cm.sum(axis=1)+cm.sum(axis=0)))
    raise ValueError('Unexpected study metric')


def verify(main,extension,ledger,output):
    panel=read(main/'completion.json')
    early=read(main/'early/frozen.json')
    arms={}
    seen={}
    checks=0
    scopes=set()
    def audit_run(folder,mode):
        receipt=read(folder/'sealed.json')
        scopes.add(receipt['scope'])
        entries=arms.setdefault(mode,dict(calls=0,tokens_in=0,tokens_out=0,returned_latency_seconds=0.,
                                         length_limited=0,known_course_rmb=0.))
        for item in receipt['calls']:
            key=item['budget_request_id']
            require(key not in seen,'One request was counted twice')
            seen[key]=(receipt['scope'],mode,item)
            entries['calls']+=1
            if item.get('course_cost_rmb') is not None:
                entries['tokens_in']+=item['input_tokens']
                entries['tokens_out']+=item['output_tokens']
                entries['returned_latency_seconds']+=item['latency_seconds']
                entries['length_limited']+=int(item.get('finish_reason')=='length')
                entries['known_course_rmb']=(entries['tokens_in']+4*entries['tokens_out'])/1e6
    for case,entry in panel['cases'].items():
        y=np.load(main/'evaluation'/(case+'.npy'),allow_pickle=False)
        for mode,arm in entry['arms'].items():
            folder=main/case/mode
            audit_run(folder,mode)
            paths={'fourth':(main/'early'/early[case][mode]['4']['file'],arm['early_sha']),
                   'final':(folder/'delivery/predictions.csv',arm['final_sha'])}
            for k,(path,fingerprint) in paths.items():
                require(sha(path)==fingerprint,'Graded output changed')
                with path.open(newline='') as stream:
                    rows=list(csv.DictReader(stream))
                require([r['row_id'] for r in rows]==list(map(str,range(len(y)))),'Graded rows out of order')
                pred=np.array([r['prediction'] for r in rows],dtype=y.dtype)
                require(abs(score_from_counts(y,pred,entry['metric'])-arm['scores'][k])<1e-12,'Independent score mismatch')
                checks+=1
    if extension:
        for c in read(extension/'completion.json')['outcomes']:
            audit_run(extension/c/'conditional-flash','conditional-flash')
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        requests=[dict(row) for row in db.execute('SELECT * FROM requests ORDER BY rowid') if row['scope'] in scopes]
        require({r['id'] for r in requests}==seen.keys(),'Untracked or missing task requests')
        transitions=[dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY seq') if r['request_id'] in seen]
        for req in requests:
            scope,mode,event=seen[req['id']]
            payload=json.loads(req['payload'])
            changes=[t['event'] for t in transitions if t['request_id']==req['id']]
            require(scope==req['scope'] and mode==payload['paired_reflection_arm']
                    and changes==['reserved',req['status']],'Task balance and settlement history differ')
            if req['status']=='known':
                settled=payload['resolution']
                require(settled['input_tokens']==event['input_tokens'] and settled['output_tokens']==event['output_tokens']
                        and req['course_units']==1000*(event['input_tokens']+4*event['output_tokens']),'Incorrect course price')
    save(main/'pending_merge.json',dict(team_merge_state='pending_team_handoff',requests=requests,transitions=transitions))
    save(output,dict(scores_independently_checked=checks,usage_by_mode=arms,ledger_requests=len(requests),
                     known_course_rmb=sum(r['course_units'] or 0 for r in requests)/1e9,
                     unresolved=sum(r['status']!='known' for r in requests),team_total_verified=False,actual_provider_debit=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('main',type=Path)
    p.add_argument('ledger',type=lambda s:Path(s).resolve())
    p.add_argument('output',type=Path)
    p.add_argument('--extension',type=Path)
    a=p.parse_args()
    verify(a.main,a.extension,a.ledger,a.output)
