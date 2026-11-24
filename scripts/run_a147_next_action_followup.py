"""Fit only different sealed A146 actions, two common internal splits, <=12 fits/0 API."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import joblib

from scripts.a_batch_common import hashes,save
from scripts.prepare_a146_grounding_pilot import RUNS
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,physical

ROOT=Path(__file__).resolve().parents[1]
SPLITS=('primary','confirmation_1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation','inputs','ledger','output','freeze'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    ledger_sha=sha(a.ledger);evaluated=read(a.evaluation/'results.json');pairs=[p for p in evaluated['pairs'] if p['complete'] and p['candidate_changed']]
    matrix={};caps={};rows=[]
    for pair in pairs:
        task=pair['task'];caps[task]=read(a.inputs/task/'capabilities.json')
        matrix.setdefault(task,set()).update(pair['candidates'].values())
    expected=sum(2*caps[t][c]['classifier_fits'] for t,names in matrix.items() for c in names)
    save(out/'protocol.json',dict(batch='A147',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A147_next_action_followup_plan_20261124_zh.md'),
        evaluation_results_sha256=sha(a.evaluation/'results.json'),different_pairs=pairs,
        unique_actions={t:sorted(v) for t,v in matrix.items()},splits=SPLITS,starting_A_fits=1636,
        maximum_classifier_fits=12,expected_classifier_fits=expected,maximum_API_requests=0,
        ledger_sha256_before=ledger_sha,maximum_seconds=900,model_timeout_seconds=180))
    try:
        if source!=read(a.freeze)['source_files']:raise ValueError('Production source differs from A145 freeze')
        if not evaluated['complete_paired_trial']:raise ValueError('A146 incomplete; no action-performance batch')
        if expected>12:raise ValueError('Whole deduplicated action matrix exceeds predeclared allowance')
        env=environment(out);os.environ.update(env)
        from scripts import a38_predict_guard
        from superagent.ab_run import Executor
        from superagent.events import RunLogger
        from superagent.metrics import MetricSpec
        from superagent.prediction_evidence import fingerprint
        inputs={}
        for task in matrix:
            folder=out/task;folder.mkdir();inputs[task]={}
            for split in SPLITS:
                original=RUNS[task]/(split+'.joblib');registration=read(RUNS[task]/(split+'_split.json'))
                if sha(original)!=registration['evaluation_payload_sha256']:raise ValueError('Producer payload changed')
                payload=joblib.load(original);metric=MetricSpec.from_payload(payload)
                state=read(a.inputs/task/'natural_state.json')
                if metric.record()!=state['task']['metric']:raise ValueError('Metric identity differs')
                if fingerprint(payload['y_search'].tolist())!=registration['label_sha256']:raise ValueError('Labels do not match registered split')
                target=folder/(split+'.joblib');shutil.copy2(original,target)
                inputs[task][split]=dict(payload_sha256=sha(target),registration=registration)
        save(out/'frozen_inputs.json',inputs)
        executors={}
        for task,names in matrix.items():
            folder=out/task;logger=RunLogger(folder)
            metric=MetricSpec.from_payload(joblib.load(folder/'primary.joblib'))
            executor=Executor(folder,caps[task],metric,start,900,180,sum(2*caps[task][c]['classifier_fits'] for c in names),logger)
            executors[task]=executor
            for split in SPLITS:
                for candidate in sorted(names):
                    if hashes(ROOT)!=source:raise ValueError('Source changed before fit')
                    os.environ['AB_A29_RUN_NAME']=task+'/'+candidate+'/'+split
                    _,facts=executor.worker(folder/(split+'.joblib'),candidate,folder/(candidate+'_'+split+'.joblib'),180)
                    if facts['status']!='completed':raise RuntimeError('Worker failed')
                    row=dict(task=task,candidate=candidate,split=split,primary_metric=metric.name,
                        search_score=facts['search_score'],search_accuracy=facts['search_accuracy'],
                        training_seconds=facts['training_seconds'],worker_wall_seconds=facts['worker_wall_seconds'],
                        classifier_fits=facts['classifier_fits'],evaluation_payload_sha256=inputs[task][split]['payload_sha256'],
                        estimator=facts['estimator'],metric_spec=facts['metric_spec'])
                    rows.append(row);save(out/'checkpoint.json',rows)
                    print(json.dumps(dict(task=task,candidate=candidate,split=split,score=row['search_score'],fits=len(physical(out)))),flush=True)
        comparisons=[]
        for pair in pairs:
            per_split=[]
            for split in SPLITS:
                scores={mode:next(r['search_score'] for r in rows if r['task']==pair['task'] and r['candidate']==candidate and r['split']==split)
                        for mode,candidate in pair['candidates'].items()}
                per_split.append(dict(split=split,scores=scores,grounded_minus_legacy=scores['grounded']-scores['legacy']))
            comparisons.append(dict(task=pair['task'],condition=pair['condition'],candidates=pair['candidates'],splits=per_split,
                mean_grounded_minus_legacy=sum(p['grounded_minus_legacy'] for p in per_split)/2))
        actual=len(physical(out));admitted=sum(e.accounting()['completed_classifier_fits'] for e in executors.values())
        if actual!=expected or admitted!=actual:raise ValueError('Physical/admitted/predeclared fit counts differ')
        if sha(a.ledger)!=ledger_sha or hashes(ROOT)!=source:raise ValueError('Ledger or source changed during offline batch')
        save(out/'results.json',dict(rows=rows,comparisons=comparisons,same_action_pairs_not_retrained=6-len(pairs),
            evidence_scope='Two internal splits per different action; seen tasks, no complete-agent or external-test comparison'))
        save(out/'completion.json',dict(status='completed' if pairs else 'skipped_same_actions',actual_classifier_fits=actual,
            real_API_requests=0,A_cumulative_fits_after=1636+actual,source_unchanged=True,ledger_unchanged=True,
            elapsed_seconds=time.monotonic()-start,results_sha256=sha(out/'results.json'),protocol_sha256=sha(out/'protocol.json')))
        print(json.dumps(dict(status='completed',actual_classifier_fits=actual,comparisons=comparisons)))
    except Exception as e:
        save(out/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=len(physical(out)),
            real_API_requests=0,A_cumulative_fits_after=1636+len(physical(out)),automatic_retry=False));raise


if __name__=='__main__':main()
