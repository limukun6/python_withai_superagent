"""Recover the interim deliverable at three fixed stopping times from existing verified trial models."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_feedback_reflection_study import independent_score
from superagent.autonomous_checkpoint import promote
from superagent.autonomous_run import planner_observation
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.predict import create_predictions


def freeze(root, out):
    outputs = {}
    for case, modes in read(root/'all_predictions_frozen.json').items():
        task = load_training_task(root/case/'train.zip',read_declarations=True)
        metric = MetricSpec.for_task(task)
        for mode in modes:
            agent = root/case/mode/'run/agent'
            hist = read(agent/'history.json')
            primary = agent/'primary.joblib'
            digest = read(agent/'primary_split.json')['payload_sha256']
            require(sha(primary)==digest,'Stored training split changed')
            payload = dict(path=primary,sha256=digest,labels=joblib.load(primary)['y_search'])
            events = [json.loads(line) for line in (agent/'events.jsonl').read_text().splitlines()]
            require(not any(e.get('kind')=='checkpoint_not_promoted' for e in events),'Incomplete historical publication')
            for cut in (3,4,5):
                observed = [r for r in hist[:cut] if r['split_id']=='primary' and r['status']=='completed']
                best = max(observed,key=lambda r:r['search_score'])
                trial = best['trial']
                model = agent/f'trial_{trial}.joblib'
                work = out/case/mode/model.stem
                predictions = work/'predictions.csv'
                if not predictions.exists():
                    work.mkdir(parents=True)
                    os.link(agent/'task_spec.json',work/'task_spec.json')
                    os.link(model,work/model.name)
                    bundle = joblib.load(model)
                    observation = planner_observation(bundle,bundle['metrics'],payload,metric)
                    require(abs(best['search_score']-observation['score'])<1e-12,'Primary evidence mismatch')
                    promote(task,metric,read(agent/'prior_selection.json'),trial,best['candidate'],
                            work/model.name,payload,observation,RunLogger(work))
                    create_predictions(str(root/case/'input.zip'),work,predictions)
                    require(read(work/'prediction_manifest.json')['delivery_status']=='partial','Not an interim model')
                outputs.setdefault(case,{}).setdefault(mode,{})[str(cut)] = dict(
                    trial=trial, candidate=best['candidate'], primary_score=best['search_score'],
                    source_model=sha(model), file=predictions.relative_to(out).as_posix(), digest=sha(predictions))
    save(out/'frozen.json',outputs)


def score(root, out, destination):
    frozen = read(out/'frozen.json')
    metrics = read(root/'protocol.json')['inputs']
    summaries = {}
    for case, arms in frozen.items():
        truth = root/'evaluation'/(case+'.npy')
        require(sha(truth)==metrics[case]['truth_sha'],'Changed development labels')
        y = np.load(truth,allow_pickle=False)
        by_cut = {}
        for cut in ('3','4','5'):
            scores, predictions = {}, {}
            for mode, checkpoints in arms.items():
                rec = checkpoints[cut]
                path = out/rec['file']
                require(sha(path)==rec['digest'],'Partial prediction changed')
                with path.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                require([r['row_id'] for r in rows]==list(map(str,range(len(y)))),'Partial row order changed')
                p = np.asarray([r['prediction'] for r in rows],dtype=y.dtype)
                predictions[mode] = p
                scores[mode] = dict(score=independent_score(metrics[case]['metric'],y,p),trial=rec['trial'],
                                    candidate=rec['candidate'],prediction_sha256=rec['digest'])
            baseline, reviewed = predictions['control'], predictions['flash']
            by_cut[cut] = dict(arms=scores,flash_changed_rows=int(sum(reviewed!=baseline)),
                               flash_corrected=int(sum((reviewed==y)&(baseline!=y))),
                               flash_spoiled=int(sum((reviewed!=y)&(baseline==y))))
        summaries[case] = dict(metric=metrics[case]['metric'],n=len(y),cutoffs=by_cut)
    save(destination,dict(status='Retrospective interruption replay on the same previously graded development panel.',
         checkpoints=[3,4,5],criterion='Actual primary best at this cutoff, earliest on ties. Re-publish through the '
         'real promotion and prediction functions. Confirmations do not replace the standing primary checkpoint.',
         new_classifier_fits=0, model_api_requests=0, tasks=summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    out=root/'interruption_replay'
    if args.freeze_only:
        freeze(root,out)
    else:
        out.mkdir()
        log=out/'unexpected_fits.jsonl'
        command=[sys.executable,'-m','scripts.audit_interrupted_reflection',str(root),str(args.output.resolve()),
                 '--weights',str(args.weights.resolve()),'--freeze-only']
        subprocess.run(command,cwd=ROOT,env=env(args.weights.resolve(),log),stdin=subprocess.DEVNULL,check=True)
        require(not log.exists() or not log.read_text().strip(),'Unexpected new fit during replay')
        score(root,out,args.output)
