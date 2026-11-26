"""Freeze and execute a new four-case, three-call semantic dialogue panel; no fitting."""
import argparse
import json
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch

from scripts.sa5_semantic_cases import grade as unused_old_grade
from scripts.sa7_multiturn_cases import build,grade
from scripts.run_sa5_semantic_panel import legacy,read,save,sha,source as prior_source


def source(root):
    result=prior_source(root)
    for name in ['scripts/sa7_multiturn_cases.py','scripts/run_sa7_multiturn.py','tests/test_multiturn_panel.py']:
        result[name]=sha(root/name)
    return result


def validate_cases(folder):
    registry=read(folder/'registry.json')
    for row in registry:
        for key,hash_key in [('package','package_sha256'),('query','query_sha256')]:
            if sha(folder/row[key])!=row[hash_key]:raise ValueError('Frozen package changed')
        if sha(folder/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Frozen oracle changed')
    return registry


def prepare(args,root):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    identity=source(root)
    save(args.output/'protocol.json',dict(batch='SA7-prepare',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA7_multiturn_plan_20261126_zh.md'),maximum_classifier_fits=0,maximum_real_API_requests=0))
    registry=build(args.cases);save(args.output/'registry.json',registry)
    from superagent.ab_run import Executor
    with (args.output/'tests.log').open('w',encoding='utf-8') as stream:
        with (patch.object(Executor,'worker',side_effect=AssertionError('No fits')),
              patch.object(socket,'create_connection',side_effect=AssertionError('No network'))):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestLoader().loadTestsFromName('tests.test_multiturn_panel'))
            controls=[]
            for row in registry:
                try:
                    task=legacy(args.cases/row['package'],None,None)
                    score=grade(task,args.cases/row['query'],read(args.cases/(row['id']+'.oracle.json')))
                    controls.append(dict(case=row['id'],status='accepted',**score))
                except Exception as error:
                    controls.append(dict(case=row['id'],status='rejected',error_type=type(error).__name__,reason=str(error)[:400]))
    unchanged=source(root)==identity
    receipt=dict(status='prepared' if result.wasSuccessful() and unchanged else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_unchanged=unchanged,local_controls=controls,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,elapsed_seconds=time.monotonic()-started,
        registry_sha256=sha(args.cases/'registry.json'),scope='new developer-authored packaging cases; no independent blind-test claim')
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False));return receipt['status']!='prepared'


class DialogueClient:
    max_requests=3
    def __init__(self,parent):
        self.parent=parent;self.start=parent.request_count;self.initial_cost=parent.spent
        self.cost_ceiling=min(parent.budget,parent.spent+.25)
    @property
    def request_count(self):return self.parent.request_count-self.start
    @property
    def blocked(self):return self.parent.blocked
    @property
    def spent(self):return self.parent.spent-self.initial_cost
    def ask(self,payload,log,role):
        if self.request_count>=self.max_requests:raise ValueError('SA7 shared three-call allowance exhausted')
        self.parent.set_deadline(time.monotonic()+90)
        return self.parent.ask(payload,log,'sa7_semantic',cost_ceiling=self.cost_ceiling)


def live(args,root):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();identity=source(root)
    save(args.output/'protocol.json',dict(batch='SA7-live',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA7_multiturn_plan_20261126_zh.md'),registry_sha256=sha(args.cases/'registry.json'),
        request_limit_per_case=3,maximum_real_API_requests=12,estimated_stop_threshold_rmb=1.,
        maximum_classifier_fits=0,validate_only=args.validate_only,automatic_retry=False,model='DeepSeek-V4.1-Flash'))
    ledger=None;parent=None;before=None;cells=[];failure=None;status='blocked_preflight'
    try:
        from superagent.task_budget import TaskBudgetLedger
        from superagent.research_budget import IncrementalResearchAccount,research_preflight
        from tools.accepted_usage_snapshot import accepted_history,load_snapshot
        from superagent.experiment_llm import ExperimentLLM
        from superagent.ab_flash import ENDPOINT,MODEL
        from superagent.events import RunLogger
        from superagent.semantic_contract import resolve_training_task
        from superagent.ab_run import Executor
        from scripts.campus_direct import campus_resolution,CAMPUS_IP
        registry=validate_cases(args.cases)
        ledger=TaskBudgetLedger(args.ledger);before=ledger.project_summary()
        if before['requests']<461 or before['pending_requests']:raise ValueError('Expected current nonempty ledger without pending requests')
        with ledger._db() as db:
            if db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(args.scope,)).fetchone()[0]:
                raise ValueError('Scope already contains calls; no automatic rerun')
        authorization=load_snapshot(args.accepted_history,sha(args.accepted_history))
        if authorization.get('allowed_scope')!=args.scope or authorization.get('registry_sha256')!=sha(args.cases/'registry.json'):
            raise ValueError('Authorization scope/registry mismatch')
        with accepted_history(args.accepted_history,sha(args.accepted_history)):
            research_preflight(ledger,args.scope,1.)
            if args.validate_only:status='validated_no_requests'
            else:
                args.private_output.mkdir(parents=True,exist_ok=False)
                parent=ExperimentLLM(ENDPOINT,args.key_file,MODEL,1.,thinking_mode='disabled',max_requests=12,
                    allowed_served_models={MODEL},account=IncrementalResearchAccount(ledger,args.scope,str(args.private_output)),streaming=True)
                status='completed'
                with (campus_resolution(CAMPUS_IP),patch.object(Executor,'worker',side_effect=AssertionError('SA7 forbids fitting'))):
                    for row in registry:
                        folder=args.private_output/row['id'];folder.mkdir();logger=RunLogger(folder)
                        client=DialogueClient(parent);clock=time.monotonic();record=dict(case=row['id'],arm='global')
                        try:
                            task=resolve_training_task(args.cases/row['package'],client,logger,mode='proactive',
                                prediction_package=args.cases/row['query'] if row['separate_query'] else None)
                            record.update(status='accepted',**grade(task,args.cases/row['query'],read(args.cases/(row['id']+'.oracle.json'))))
                        except Exception as error:
                            record.update(status='rejected',error_type=type(error).__name__,reason=str(error)[:650])
                        record.update(real_API_requests=client.request_count,known_course_cost_rmb=client.spent,
                                      elapsed_seconds=time.monotonic()-clock)
                        cells.append(record);save(args.output/'cells.json',cells)
                        print(json.dumps(dict(case=row['id'],status=record['status'],complete=record.get('complete'),requests=client.request_count)),flush=True)
                        if parent.blocked:status='stopped_new_unresolved';break
        if source(root)!=identity:raise RuntimeError('Frozen source changed during panel')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error)[:700])
    receipt=dict(status=status,error=failure,cells=cells,actual_classifier_fits=0,
        real_API_requests=parent.request_count if parent else 0,known_incremental_course_cost_rmb=parent.spent if parent else 0,
        project_before=before,project_after=ledger.project_summary() if ledger else None,
        elapsed_seconds=time.monotonic()-started,team_total_verified=False,actual_school_debit_rmb=None,
        scope='bounded dialogue on new developer-authored cases; no quality or real-task generality claim')
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(dict(status=status,real_API_requests=receipt['real_API_requests'],error=failure),ensure_ascii=False))
    return status not in ('completed','validated_no_requests') or failure is not None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','live'])
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private-output',type=Path);parser.add_argument('--ledger',type=Path)
    parser.add_argument('--accepted-history',type=Path);parser.add_argument('--key-file',type=Path)
    parser.add_argument('--scope',default='sa7_multiturn_20261126');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.stage=='prepare':return prepare(args,root)
    if not args.ledger or not args.accepted_history:parser.error('live needs the real ledger and accepted-history snapshot')
    if not args.validate_only and (not args.key_file or not args.private_output):parser.error('live needs key-file and private-output')
    return live(args,root)


if __name__=='__main__':raise SystemExit(main())
