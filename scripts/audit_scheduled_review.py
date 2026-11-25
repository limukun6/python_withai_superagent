"""Replay an arithmetic comparator and reconcile all paid receipts for both review studies."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_conditional_review_study import get_predictions
from scripts.run_scheduled_review_study import CASES, FRESH, MODE, reload_pointer
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_conditional_study_receipts import score_from_counts
from superagent.autonomous_checkpoint import promote
from superagent.autonomous_run import planner_observation
from superagent.conditional_review import compare, evaluate_probe
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec
from superagent.predict import create_predictions


PLAN=ROOT/'research/conditional_review_1125/scheduled_ablation_protocol.json'


def arithmetic(root):
    frozen=read(root/'frozen.json')
    result={}
    for case,paired in frozen.items():
        if 'interim' not in paired:
            result[case]=dict(eligible=False)
            continue
        folder=root/case
        agent=folder/MODE/'run/agent'
        answer=read(agent/'scheduled_review.json')
        packet=read(agent/'scheduled_review_input.json')
        require(identity(packet)==answer['input_digest'],'Pre-measurement evidence changed')
        probe=min(packet['options'],key=lambda o:(o['required_delta'],o['challenger']))
        ref,name=packet['incumbent'],probe['challenger']
        fold=probe['third_fold']
        history=read(agent/'history.json')[:answer['trial']]
        task=load_training_task(folder/'train.zip',read_declarations=True)
        metric=MetricSpec.for_task(task)
        def observation(f,n):
            number=next(r['trial'] for r in history if r['status']=='completed' and r['split_id']==f and r['candidate']==n)
            artifact=agent/f'trial_{number}.joblib'
            bundle=joblib.load(artifact)
            path=agent/(f+'.joblib')
            payload=dict(path=path,labels=joblib.load(path)['y_search'],sha256=read(agent/(f+'_split.json'))['payload_sha256'])
            require(sha(path)==payload['sha256'],'Split contents changed')
            return planner_observation(bundle,bundle['metrics'],payload,metric),number,artifact,payload
        latest=compare(fold,observation(fold,ref)[0],observation(fold,name)[0])
        gain,passed=evaluate_probe(probe,latest,probe['required_delta'])
        output=root/'arithmetic'/case
        if passed:
            obs,number,model,payload=observation('primary',name)
            output.mkdir(parents=True,exist_ok=True)
            if not (output/model.name).exists():
                os.link(model,output/model.name)
                os.link(agent/'task_spec.json',output/'task_spec.json')
            promote(task,metric,read(agent/'prior_selection.json'),number,name,output/model.name,payload,obs,RunLogger(output))
            path=output/'predictions.csv'
            create_predictions(str(folder/'input.zip'),output,path)
        else:
            path=reload_pointer(output,agent,'scheduled_selection_before.json',folder/'input.zip')
        require(read(output/'prediction_manifest.json')['delivery_status']=='partial','Arithmetic checkpoint could not reload')
        result[case]=dict(eligible=True,chosen_option=name,trial=answer['trial'],passes=passed,weighted_delta=gain,
                          third=latest,file=str(path.relative_to(root)),digest=sha(path))
    save(root/'arithmetic_frozen.json',result)


def audit(a):
    out=a.root
    panel=read(out/'completion.json')
    frozen=read(out/'frozen.json')
    rule=read(out/'arithmetic_frozen.json')
    inputs=read(out/'protocol.json')['inputs']
    comparison={}
    for case in CASES:
        rec=rule[case]
        if not rec['eligible']:
            comparison[case]=dict(eligible=False)
            continue
        p=get_predictions(out/rec['file'],np.zeros(inputs[case]['outer'],dtype=int),rec['digest'])
        partial=frozen[case]['interim']
        preds={k:get_predictions(out/r['file'],p,r['digest']) for k,r in partial.items()}
        item=dict(eligible=True,arithmetic_adopts=rec['passes'],weighted_delta=rec['weighted_delta'],trial=rec['trial'],
                  live_action=panel['records'][case]['review']['executed'],different_from_live=int(sum(p!=preds['after'])),
                  different_from_control=int(sum(p!=preds['control'])))
        if case in FRESH:
            truth=out/'evaluation'/(case+'.npy')
            require(sha(truth)==inputs[case]['truth_sha'],'Outer labels changed after freezing')
            y=np.load(truth,allow_pickle=False)
            scores=[score_from_counts(y,preds[k],'balanced_accuracy') for k in ('control','after')]
            require(np.allclose(scores,panel['records'][case]['interim_scores'],atol=1e-12,rtol=0),'Independent interim score mismatch')
            item.update(arithmetic_score=score_from_counts(y,p,'balanced_accuracy'),control_score=scores[0],live_score=scores[1])
        comparison[case]=item
    known={}
    scopes=set()
    for root in (a.main,a.previous,out):
        for path in root.glob('*/*/sealed.json'):
            receipt=read(path)
            scopes.add(receipt['scope'])
            for event in receipt['calls']:
                key=event['budget_request_id']
                require(key not in known,'Duplicate budget request')
                known[key]=receipt['scope'],event
    with sqlite3.connect(a.ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM requests ORDER BY rowid') if r['scope'] in scopes]
        require({r['id'] for r in rows}==known.keys(),'Unaccounted request in a canonical scope')
        transitions=[dict(t) for t in db.execute('SELECT * FROM transitions ORDER BY seq') if t['request_id'] in known]
    for row in rows:
        scope,event=known[row['id']]
        require(scope==row['scope'] and row['status']=='known','Unsettled or different task')
        settled=json.loads(row['payload'])['resolution']
        counts=[event[k] for k in ('input_tokens','output_tokens')]
        require(counts==[settled[k] for k in ('input_tokens','output_tokens')] and row['course_units']==1000*(counts[0]+4*counts[1])
                and [t['event'] for t in transitions if t['request_id']==row['id']]==['reserved','known'],'Settlement mismatch')
    save(out/'pending_merge.json',dict(team_merge_state='pending_team_handoff',requests=rows,transitions=transitions))
    public=dict(protocol_sha256=sha(PLAN),comparison=comparison,new_fits=0,new_api_calls=0,team_ledger_merge='pending',
                combined_paid_calls=len(rows),combined_input_tokens=sum(e['input_tokens'] for _,e in known.values()),
                combined_output_tokens=sum(e['output_tokens'] for _,e in known.values()),
                combined_course_rmb=sum(r['course_units'] for r in rows)/1e9,unresolved=0,actual_provider_debit=None)
    save(a.output,public)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=lambda x:Path(x).resolve())
    for key in ('previous','main','weights','ledger','output'):
        parser.add_argument('--'+key,type=lambda x:Path(x).resolve(),required=True)
    a=parser.parse_args()
    if os.getenv('AB_B1_OFFLINE')=='1':
        arithmetic(a.root)
    else:
        fitlog=a.root/'unexpected_arithmetic_fits.jsonl'
        subprocess.run([sys.executable,'-m','scripts.audit_scheduled_review',*sys.argv[1:]],cwd=ROOT,
                       env=env(a.weights,fitlog),check=True)
        require(not fitlog.exists() or not fitlog.read_text().strip(),'Arithmetical comparison performed an unexpected fit')
        audit(a)
