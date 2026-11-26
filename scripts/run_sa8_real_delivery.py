"""Prepare and execute three bounded, real-data autonomous delivery paths without opening answers until all paths terminate."""
import argparse
from contextlib import redirect_stderr,redirect_stdout
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa7_multiturn import DialogueClient
from scripts.sa7_multiturn_cases import grade
from scripts.sa8_real_delivery_cases import build


def source(root):
    names=[p.relative_to(root).as_posix() for p in sorted((root/'superagent').glob('*.py'))]
    names+=['unified_agent.py','scripts/run_sa8_real_delivery.py','scripts/sa8_real_delivery_cases.py',
            'scripts/sa7_multiturn_cases.py','tests/test_real_delivery_cases.py']
    return {name:sha(root/name) for name in names}


def validate_registry(cases):
    registry=read(cases/'registry.json')
    for row in registry:
        for field,hashfield in [('package','package_sha256'),('query','query_sha256')]:
            if sha(cases/row[field])!=row[hashfield]:raise ValueError('Frozen real task changed')
        if sha(cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Frozen oracle changed')
    return registry


def prepare(args,root):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();identity=source(root)
    save(args.output/'protocol.json',dict(batch='SA8-prepare',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA8_real_delivery_plan_20261126_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0))
    registry=build(args.cases);save(args.output/'registry.json',registry)
    from superagent.ab_run import Executor
    with (args.output/'tests.log').open('w',encoding='utf-8') as stream:
        with (patch.object(Executor,'worker',side_effect=AssertionError('Preparation forbids training')),
              patch.object(socket,'create_connection',side_effect=AssertionError('Preparation forbids network'))):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestLoader().loadTestsFromName('tests.test_real_delivery_cases'))
    receipt=dict(status='prepared' if result.wasSuccessful() and source(root)==identity else 'failed',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        registry_sha256=sha(args.cases/'registry.json'),elapsed_seconds=time.monotonic()-started)
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(receipt));return receipt['status']!='prepared'


class DeliveryClient(DialogueClient):
    def set_deadline(self,deadline):self.task_deadline=deadline
    def ask(self,payload,log,role):
        if self.request_count>=self.max_requests:raise ValueError('SA8 shared three-call allowance exhausted')
        self.parent.set_deadline(min(getattr(self,'task_deadline',float('inf')),time.monotonic()+90))
        return self.parent.ask(payload,log,'sa8_semantic',cost_ceiling=self.cost_ceiling)


def run(args,root):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();identity=source(root)
    save(args.output/'protocol.json',dict(batch='SA8-live',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA8_real_delivery_plan_20261126_zh.md'),registry_sha256=sha(args.cases/'registry.json'),
        fit_limit_per_path=4,maximum_classifier_fits=12,maximum_real_API_requests=9,request_limit_per_path=3,
        estimated_stop_threshold_rmb=.75,wall_per_path=240,seed=18643,
        oracle_gate='predeclared evaluator at standby; accepts/rejects only, never repairs or chooses algorithms',
        scoring='open development answers only after all attempted paths are terminal and predictions are sealed',validate_only=args.validate_only))
    ledger=None;parent=None;before=None;results=[];failure=None;status='blocked_preflight';total_fits=0
    try:
        from superagent.task_budget import TaskBudgetLedger
        from superagent.research_budget import IncrementalResearchAccount,research_preflight
        from tools.accepted_usage_snapshot import accepted_history,load_snapshot
        from superagent.experiment_llm import ExperimentLLM
        from superagent.ab_flash import ENDPOINT,MODEL
        from scripts.campus_direct import campus_resolution,CAMPUS_IP
        from superagent.autonomous_checkpoint import standby as actual_standby
        from superagent.metrics import MetricSpec
        from unified_agent import train as official_train
        registry=validate_registry(args.cases)
        previous=read(root/'research/sa7_live_01/results.json')
        if previous['status']!='completed' or not any(r.get('complete') for r in previous['cells']):
            raise ValueError('SA7 has not met the preregistered delivery gate')
        ledger=TaskBudgetLedger(args.ledger);before=ledger.project_summary()
        if before['pending_requests'] or before['unresolved_requests']!=16:raise ValueError('New unresolved/pending usage blocks SA8')
        with ledger._db() as db:
            if db.execute('SELECT 1 FROM requests WHERE scope=?',(args.scope,)).fetchone():raise ValueError('No automatic reuse of an attempted scope')
        authorization=load_snapshot(args.accepted_history,sha(args.accepted_history))
        if authorization.get('allowed_scope')!=args.scope or authorization.get('registry_sha256')!=sha(args.cases/'registry.json'):
            raise ValueError('Authorization scope/case registry changed')
        with accepted_history(args.accepted_history,sha(args.accepted_history)):
            research_preflight(ledger,args.scope,.75)
            if args.validate_only:status='validated_no_requests'
            else:
                args.private_output.mkdir(parents=True,exist_ok=False)
                parent=ExperimentLLM(ENDPOINT,args.key_file,MODEL,.75,thinking_mode='disabled',max_requests=9,
                    allowed_served_models={MODEL},account=IncrementalResearchAccount(ledger,args.scope,str(args.private_output)),streaming=True)
                status='completed';os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                with campus_resolution(CAMPUS_IP):
                    for row in registry:
                        client=DeliveryClient(parent);clock=time.monotonic();run_path=args.private_output/row['id']
                        oracle=read(args.cases/(row['id']+'.oracle.json'));gate_results=[]
                        def checked_standby(task,*rest):
                            verification=grade(task,args.cases/row['query'],oracle);gate_results.append(verification)
                            if not verification['complete']:raise RuntimeError('Predetermined semantic oracle gate failed: '+repr(verification))
                            return actual_standby(task,*rest)
                        result=dict(case=row['id'],status='failed',complete=False)
                        try:
                            with (patch('superagent.unified.configured_flash',return_value=client),
                                  patch('superagent.autonomous_run.standby',side_effect=checked_standby)):
                                code=official_train(args.cases/row['package'],run_path,fit_limit=4,time_limit=240,
                                    self_review='protected',feedback_reflection='opening-local',task_interpretation='proactive',
                                    prediction_input=args.cases/row['query'])
                            if code:raise RuntimeError('Official training exit '+str(code))
                            selection=read(run_path/'agent/final_selection.json')
                            if not selection.get('refit'):raise RuntimeError('Partial model is not full delivery')
                            destination=run_path/'fresh/predictions.csv'
                            environment=os.environ.copy()
                            for key in list(environment):
                                if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_'):environment.pop(key)
                            proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str(args.cases/row['query']),
                                '--run-dir',str(run_path),'--output',str(destination)],cwd=root,env=environment,
                                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
                            (args.private_output/(row['id']+'_predict.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
                            if proc.returncode:raise RuntimeError('Fresh prediction failed; see private log')
                            recipe=oracle['expected']['output']
                            with destination.open(encoding=recipe['encoding'],newline='') as stream:
                                records=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
                            if len(records)!=row['prediction_rows'] or list(records[0])!=[c['name'] for c in recipe['columns']]:
                                raise ValueError('Fresh export rows/header differ from oracle')
                            if [r[recipe['columns'][0]['name']] for r in records]!=[f'{i:06d}' for i in range(len(records))]:
                                raise ValueError('Original IDs/order changed')
                            if recipe['encoding']=='utf-8-sig' and not destination.read_bytes().startswith(b'\xef\xbb\xbf'):
                                raise ValueError('Required BOM absent')
                            if row['id']=='V1':
                                scores=np.array([float(r['PositiveScore']) for r in records])
                                if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Invalid exported probabilities')
                            result.update(status='delivered',complete=True,model_sha256=selection['model_sha256'],
                                family=selection['family'],refit=True,prediction_sha256=sha(destination),prediction_rows=len(records))
                        except Exception as error:
                            result.update(error=dict(type=type(error).__name__,message=str(error)[:900]))
                        fit_file=run_path/'agent/classifier_fits.json'
                        fits=read(fit_file)['attempted_classifier_fits'] if fit_file.is_file() else 0
                        total_fits+=fits
                        if fits>4 or total_fits>12:raise RuntimeError('Physical fit budget exceeded')
                        result.update(actual_classifier_fits=fits,real_API_requests=client.request_count,
                            known_course_cost_rmb=client.spent,semantic_gate=gate_results,elapsed_seconds=time.monotonic()-clock)
                        results.append(result);save(args.output/'deliveries_before_scoring.json',results)
                        print(json.dumps(dict(case=row['id'],status=result['status'],fits=fits,requests=client.request_count)),flush=True)
                        if parent.blocked:status='stopped_new_unresolved';break
                # All attempted paths are now terminal. Freeze every existing delivery
                # before opening any held development labels.
                save(args.output/'sealed_deliveries.json',[dict(case=r['case'],model=r.get('model_sha256'),predictions=r.get('prediction_sha256')) for r in results])
                for result in results:
                    if not result['complete']:continue
                    row=next(r for r in registry if r['id']==result['case']);oracle=read(args.cases/(row['id']+'.oracle.json'))
                    path=args.private_output/row['id']/'fresh/predictions.csv'
                    if sha(path)!=result['prediction_sha256']:raise ValueError('Sealed prediction changed')
                    answer_file=args.cases/(row['id']+'.development_answers.npy')
                    if sha(answer_file)!=row['development_answers_sha256']:raise ValueError('Development answer changed')
                    truth=np.load(answer_file,allow_pickle=False)
                    recipe=oracle['expected']['output'];column=next(c for c in recipe['columns'] if c['kind']=='label')
                    with path.open(encoding=recipe['encoding'],newline='') as stream:
                        records=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
                    inverse={str(v):k for k,v in column.get('mapping',{}).items()}
                    predicted=np.array([inverse.get(r[column['name']],r[column['name']]) for r in records])
                    metric=MetricSpec('macro_f1',tuple(oracle['expected']['classes']))
                    result['development_score']=metric.evaluate(truth,predicted)
                    result['development_correct']=int(sum(truth==predicted))
                if status=='completed' and not all(r['complete'] for r in results):status='completed_with_failures'
        if source(root)!=identity:raise RuntimeError('Production source changed during SA8')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error)[:1000])
    receipt=dict(status=status,error=failure,cases=results,actual_classifier_fits=total_fits,
        real_API_requests=parent.request_count if parent else 0,known_incremental_course_cost_rmb=parent.spent if parent else 0,
        project_before=before,project_after=ledger.project_summary() if ledger else None,
        elapsed_seconds=time.monotonic()-started,team_total_verified=False,actual_school_debit_rmb=None,
        scope='small real-data engineering deliveries; SMS previously used; no matched quality improvement or unseen guarantee')
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(dict(status=status,fits=total_fits,API=receipt['real_API_requests'],error=failure),ensure_ascii=False))
    return status not in ('completed','validated_no_requests') or failure is not None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private-output',type=Path)
    parser.add_argument('--ledger',type=Path);parser.add_argument('--accepted-history',type=Path);parser.add_argument('--key-file',type=Path)
    parser.add_argument('--scope',default='sa8_real_delivery_20261126');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.stage=='prepare':return prepare(args,root)
    if not args.ledger or not args.accepted_history:parser.error('run needs real ledger and history authorization')
    if not args.validate_only and (not args.key_file or not args.private_output):parser.error('live run needs key-file and private-output')
    return run(args,root)


if __name__=='__main__':raise SystemExit(main())
