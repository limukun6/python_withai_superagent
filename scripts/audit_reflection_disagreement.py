"""Run the predeclared third-fold follow-up using the existing protected incumbent and candidate."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_feedback_reflection_study import independent_score
from superagent.ab_run import Executor, registered_capabilities
from superagent.autonomous_run import freeze_model, installed_methods, planner_observation
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.open_methods import capability
from superagent.predict import create_predictions


PLAN = ROOT/'research/feedback_reflection_1125/third_fold_protocol.json'


def follow(root, output):
    panel = read(root/'all_predictions_frozen.json')
    records = {}
    study_fits = 0
    for case in panel:
        base = root/case/'control'
        agent = base/'run/agent'
        stop = read(agent/'stop_review.json')
        old = stop['comparisons']
        deltas = [r['delta'] for r in old]
        if len(deltas)!=2 or not min(deltas)<0<max(deltas):
            records[case] = dict(triggered=False,reason='No opposite signs on the first two folds')
            continue
        challenger = stop['candidate']
        history = read(agent/'history.json')
        require(len(stop['incumbent'])==1,'Single archived incumbent required')
        leader = stop['incumbent'][0]
        folds = sorted({p.name.removesuffix('_split.json') for p in agent.glob('confirmation_*_split.json')}
                       - {r['fold'] for r in old})
        require(len(folds)==1,'Third fold ambiguous')
        fold = folds[0]
        previous = {r['candidate']:r for r in history if r['split_id']==fold and r['status']=='completed'}
        require(leader in previous and challenger not in previous,'Third-fold evidence differs from registration')
        partition = read(agent/(fold+'_split.json'))
        held = set(partition['held'])
        for r in old:
            require(held.isdisjoint(read(agent/(r['fold']+'_split.json'))['held']),'Evaluation rows overlap')
        data = agent/(fold+'.joblib')
        require(sha(data)==partition['payload_sha256'],'Partition changed')
        labels = joblib.load(data)['y_search']
        payload = dict(path=data,labels=labels,sha256=sha(data))
        work = output/case
        work.mkdir()
        latest = read(agent/'decisions.json')[-1]
        require(sha(agent/latest['registry'])==latest['registry_sha256'],'Registry changed')
        for a,b in [(latest['registry'],'candidate_registry.json'),('task_spec.json','task_spec.json'),
                    ('refit_data.joblib','refit_data.joblib')]:
            os.link(agent/a,work/b)
        task = load_training_task(root/case/'train.zip',read_declarations=True)
        spec, metric = read(agent/'task_contract.json'), MetricSpec.for_task(task)
        entries=read(work/'candidate_registry.json')['entries']
        if challenger in entries:
            cap=capability(entries[challenger])
        else:
            width=task.X_train.shape[1] if task.kind=='numeric_tabular' else None
            cap=registered_capabilities(spec,len(task.y_train),numeric_kernel_width=width)[challenger]
        completed_fits=read(base/'sealed.json')['fits']
        limit=min(16-completed_fits,2-study_fits)
        require(cap['classifier_fits']+cap['refit_classifier_fits']<=limit,'Study fit allowance exhausted')
        log=RunLogger(work)
        executor=Executor(work,{challenger:cap},metric,time.monotonic(),180.,120.,limit,log)
        incumbent_bundle=joblib.load(agent/f"trial_{previous[leader]['trial']}.joblib")
        require(incumbent_bundle['metrics']['resolved_configuration']['id']==leader,'Incumbent identity changed')
        before=planner_observation(incumbent_bundle,incumbent_bundle['metrics'],payload,metric)
        result=work/'third_challenger.joblib'
        _,facts=executor.worker(data,challenger,result,120.,reserve=cap['refit_classifier_fits'])
        after=planner_observation(joblib.load(result),facts,payload,metric)
        require(before['row_key']==after['row_key'],'Rows differ between contestants')
        a,b=before['predictions']==labels,after['predictions']==labels
        third=dict(fold=fold,rows=len(labels),incumbent=before['score'],challenger=after['score'],
                   delta=after['score']-before['score'],gained=int(sum(~a&b)),lost=int(sum(a&~b)))
        comparisons=[*old,third]
        margin=np.average([r['delta'] for r in comparisons],weights=[r['rows'] for r in comparisons])
        use_new=third['delta']>0 and sum(r['delta']>0 for r in comparisons)>=2 and margin>=.003
        if use_new:
            choice=dict(action='stop',members=[challenger],utility=np.mean([r['challenger'] for r in comparisons]),
                        evidence=dict(scores={r['fold']:r['challenger'] for r in comparisons}))
            selected=freeze_model(task,spec,metric,choice,executor,work,log,120.,challenger=True)
        else:
            selected=read(agent/'incumbent_selection.json')
            require(sha(agent/selected['model_file'])==stop['incumbent_model'],'Incumbent artifact changed')
            os.link(agent/selected['model_file'],work/selected['model_file'])
        save(work/'final_selection.json',selected)
        pred=create_predictions(str(root/case/'input.zip'),work,work/'delivery/predictions.csv')
        counts=executor.accounting()
        require(counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=limit and
                not counts['unknown_failed_attempts'],'Physical accounting incomplete')
        study_fits+=counts['attempted_classifier_fits']
        records[case]=dict(triggered=True,incumbent=leader,challenger=challenger,comparisons=comparisons,
                           weighted_delta=float(margin),adopted=bool(use_new),previous_fits=completed_fits,
                           cumulative_fits=completed_fits+counts['attempted_classifier_fits'],
                           selected_model=selected['model_sha256'],predictions=sha(pred))
    save(output/'frozen.json',dict(protocol_sha256=sha(PLAN),cases=records,extra_fits=study_fits))


def evaluate(root, output, destination):
    report=read(output/'frozen.json')
    count=output/'fits.jsonl'
    require(report['extra_fits']==(len(count.read_text().splitlines()) if count.exists() else 0)<=2,'Unmatched fit accounting')
    metrics=read(root/'protocol.json')['inputs']
    for case,r in report['cases'].items():
        if not r['triggered']:
            continue
        path=output/case/'delivery/predictions.csv'
        baseline=root/case/'control/delivery/predictions.csv'
        truth=root/'evaluation'/(case+'.npy')
        require(sha(path)==r['predictions'] and sha(truth)==metrics[case]['truth_sha'],'Frozen output or truth changed')
        y=np.load(truth,allow_pickle=False)
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        require([x['row_id'] for x in rows]==list(map(str,range(len(y)))),'Row order changed')
        p=np.asarray([x['prediction'] for x in rows],dtype=y.dtype)
        r.update(heldout_score=independent_score(metrics[case]['metric'],y,p),heldout_rows=len(y),
                 same_predictions_as_control=sha(path)==sha(baseline))
    report.update(api_calls=0,course_test_accessed=False,scope='Training-fold follow-up on a previously graded development panel.')
    save(destination,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--train-only',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    output=root/'third_fold'
    if args.train_only:
        with installed_methods():
            follow(root,output)
    else:
        output.mkdir()
        settings=env(args.weights.resolve(),output/'fits.jsonl')
        command=[sys.executable,'-m','scripts.audit_reflection_disagreement',str(root),str(args.destination.resolve()),
                 '--weights',str(args.weights.resolve()),'--train-only']
        subprocess.run(command,cwd=ROOT,env=settings,stdin=subprocess.DEVNULL,check=True)
        evaluate(root,output,args.destination)
