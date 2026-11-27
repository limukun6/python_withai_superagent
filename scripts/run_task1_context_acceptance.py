"""Seal an autonomous Task1 prediction before a single designated-validation score."""
import argparse
import csv
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.run_task1_profile_screen import load_split, save
from scripts.unified_released_study import ROOT, env, make_zip, read, require, sha
from scripts.verify_anonymous_planner import execute
from superagent.ab_methods import CONTEXT, context_candidate


def run(task, out, private):
    out.mkdir(parents=True,exist_ok=False)
    private.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    physical = private/'physical.jsonl'
    access = private/'members.jsonl'
    fixed_fits = 0
    name = 'in_vehicle_coupon_recommendation/'
    try:
        require(read(ROOT/'research/task1_joint_context_01/completion.json')['promoted']=='tokens','Screen did not qualify')
        allowed = [name+s for s in ('C_train.npy','y_train.npy','Readme.md','info.json')]
        save(out/'protocol.json',dict(source=source_identity(ROOT),runner=sha(Path(__file__)),
            plan=sha(ROOT/'docs/batches/TASK1_CONTEXT_AGENT_PLAN_20261127_zh.md'),
            raw=sha(task),agent_fits=10,fixed_diagnostic_fits=1,api_calls=0,allowed_members=allowed))
        features = private/'features.zip'
        with zipfile.ZipFile(task) as archive:
            query_bytes = archive.read(name+'C_val.npy')
        make_zip(features,[('C_test.npy',query_bytes)])
        settings = env(private/'unused_weights',physical)
        for key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID',
                    'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX'):
            settings.pop(key,None)
        settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))),PYTHONUTF8='1',
                        UNIFIED_RAW_INPUT=str(task),UNIFIED_RAW_MEMBERS=json.dumps(allowed),UNIFIED_RAW_LOG=str(access))
        run_dir = private/'agent_run'
        output = private/'delivery/predictions.csv'
        execute(['solve',task,'--prediction-input',features,'--run-dir',run_dir,'--fit-limit',10,'--output',output],
                private/'solve.log',settings)
        agent = run_dir/'agent'
        outcome = read(agent/'run_outcome.json')
        selected = read(agent/'final_selection.json')
        decisions = read(agent/'decisions.json')
        counts = len(physical.read_text().splitlines())
        members = decisions[-1]['choice']['members']
        require(read(output.parent/'prediction_manifest.json')['delivery_phase']=='selected' and
                outcome['status']=='completed' and outcome['attempted_classifier_fits']==counts<=10 and
                outcome['new_api_requests']==0 and not outcome['formal_test_accessed'], 'Incomplete solution or accounting')
        opened = sorted(set(json.loads(s) for s in access.read_text(encoding='utf-8').splitlines()))
        require({name+'C_train.npy',name+'y_train.npy'}<=set(opened)<=set(allowed),'Unexpected training member access')
        require(selected['refit'] and selected['model_sha256']==sha(agent/selected['model_file']) and
                decisions[-1]['choice']['action']=='stop','Model freeze could not be verified')
        replay = private/'replay/predictions.csv'
        execute(['predict',features,'--run-dir',run_dir,'--output',replay],private/'reload.log',settings)
        require(sha(output)==sha(replay) and counts==len(physical.read_text().splitlines()),'Reload did not match')
        frozen = {'agent':output}
        if members != [CONTEXT]:
            x,y,_ = load_split(task,'train')
            q = np.load(io.BytesIO(query_bytes),allow_pickle=True)
            fixed_fits = 1
            _,pred = context_candidate().fit_predict(x,y,q)
            path = private/'diagnostic.csv'
            with path.open('x',encoding='utf-8',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['row_id','prediction'])
                writer.writerows(enumerate(np.asarray(pred).reshape(-1).tolist()))
            frozen['fixed_diagnostic'] = path
        seal = dict(predictions={k:sha(p) for k,p in frozen.items()},model=selected['model_sha256'],
            members=members,opened=opened,profile=read(agent/'training_profile.json').get('recurring_profile_columns'),
            actions=[d['choice']['action'] for d in decisions],
            workers=[json.loads(s) for s in physical.read_text().splitlines()],
            source=source_identity(ROOT),independent_reload=True)
        require(seal['source']==read(out/'protocol.json')['source'],'Source changed during execution')
        save(out/'sealed.json',seal)
        with zipfile.ZipFile(task) as archive:
            truth = np.load(io.BytesIO(archive.read(name+'y_val.npy')),allow_pickle=False).reshape(-1)
        scores = {}
        for k,p in frozen.items():
            require(sha(p)==seal['predictions'][k],'Changed prediction')
            with p.open(encoding='utf-8-sig',newline='') as f:
                rows = list(csv.DictReader(f))
            require([r['row_id'] for r in rows]==list(map(str,range(len(truth)))),'Incorrect row order')
            correct = int(np.sum(np.asarray([int(r['prediction']) for r in rows])==truth))
            scores[k] = dict(correct=correct,total=len(truth),accuracy=correct/len(truth))
        save(out/'completion.json',dict(status='complete',scores=scores,agent_fits=counts,diagnostic_fits=fixed_fits,
             api_calls=0,tokens=0,course_yuan=0,seconds=round(time.monotonic()-start,3),official_test_opened=False))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),seconds=round(time.monotonic()-start,3),
             observed_worker_fits=len(physical.read_text().splitlines()) if physical.exists() else 0,
             diagnostic_fit_attempts=fixed_fits,api_calls=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('task','output','private'):
        parser.add_argument('--'+flag,type=lambda p:Path(p).resolve(),required=True)
    args=parser.parse_args()
    run(args.task,args.output,args.private)
