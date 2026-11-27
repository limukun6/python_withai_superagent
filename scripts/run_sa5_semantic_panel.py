"""Prepare or run the fixed SA5 semantic panel; never fit classifiers or retry failed requests."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch

from scripts.sa5_semantic_cases import build,grade


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def legacy(package,client,log):
    from superagent.data import load_training_task
    from superagent.format_intake import DECODE_ERRORS,NativeContract,adapt_training
    try: return load_training_task(package,read_declarations=True)
    except DECODE_ERRORS as native:
        try: return adapt_training(package,client,log)
        except NativeContract: raise native


def source(root):
    paths=['superagent/semantic_contract.py','superagent/format_intake.py','superagent/output_recipe.py',
           'superagent/format_guidance.py','superagent/experiment_llm.py','superagent/research_budget.py',
           'tools/accepted_usage_snapshot.py','scripts/sa5_semantic_cases.py','scripts/run_sa5_semantic_panel.py',
           'tests/test_semantic_panel.py']
    return {p:sha(root/p) for p in paths}


def prepare(args,root):
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',dict(stage='SA5-prepare',source_sha256=source(root),
        plan_sha256=sha(root/'docs/batches/SA5_semantic_panel_plan_20261126_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,scope='development cases, not independent human gold labels'))
    started=time.monotonic(); registry=build(args.cases)
    save(args.output/'case_registry.json',registry)
    from superagent.ab_run import Executor
    with (args.output/'tests.log').open('w',encoding='utf-8') as stream:
        with (patch.object(Executor,'worker',side_effect=AssertionError('No fitting')),
              patch.object(socket,'create_connection',side_effect=AssertionError('No networking'))):
            suite=unittest.TestLoader().loadTestsFromName('tests.test_semantic_panel')
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            baseline=[]
            for row in registry:
                try:
                    task=legacy(args.cases/row['package'],None,None)
                    record=grade(task,args.cases/row['package'],read(args.cases/(row['id']+'.oracle.json')))
                    baseline.append(dict(case=row['id'],status='accepted',**record))
                except Exception as error:
                    baseline.append(dict(case=row['id'],status='rejected',error_type=type(error).__name__,reason=str(error)[:300]))
    receipt=dict(status='prepared' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),legacy_no_client=baseline,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,elapsed_seconds=time.monotonic()-started,
        registry_sha256=sha(args.cases/'registry.json'),scope='oracle/preflight checks only; no real LLM success claim')
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False));return not result.wasSuccessful()


class CellClient:
    """One request per cell, charged to the same eight-request parent and real account."""
    max_requests=1
    def __init__(self,parent,arm): self.parent,self.arm,self.start=parent,arm,parent.request_count
    @property
    def request_count(self): return self.parent.request_count-self.start
    @property
    def blocked(self): return self.parent.blocked
    def ask(self,payload,log,role):
        if self.request_count>=1: raise ValueError('SA5 single-turn cell allowance exhausted')
        self.parent.set_deadline(time.monotonic()+90)
        return self.parent.ask(payload,log,'sa5_'+self.arm,cost_ceiling=min(self.parent.budget,self.parent.spent+.10))


def live(args,root):
    args.output.mkdir(parents=True,exist_ok=False)
    registry=read(args.cases/'registry.json'); identity=source(root)
    protocol=dict(stage='SA5-live',source_sha256=identity,registry_sha256=sha(args.cases/'registry.json'),
        maximum_classifier_fits=0,maximum_real_API_requests=8,estimated_stop_threshold_rmb=.80,
        request_limit_per_cell=1,model='DeepSeek-V4.1-Flash',automatic_retry=False,validate_only=args.validate_only,
        plan_sha256=sha(root/'docs/batches/SA5_semantic_panel_plan_20261126_zh.md'),
        B_new_usage_merge_state='published aggregates; per-request original outbox not present locally; team total unverified')
    save(args.output/'protocol.json',protocol)
    started=time.monotonic(); parent=None; ledger=None; before=None; cells=[]; failure=None
    status='blocked_preflight'; unknowns=[]
    try:
        from superagent.task_budget import TaskBudgetLedger
        from superagent.research_budget import IncrementalResearchAccount,research_preflight
        from tools.accepted_usage_snapshot import accepted_history,row_digest,load_snapshot
        from superagent.experiment_llm import ExperimentLLM
        from superagent.ab_flash import ENDPOINT,MODEL
        from superagent.events import RunLogger
        from superagent.semantic_contract import resolve_training_task
        from scripts.campus_direct import campus_resolution,CAMPUS_IP
        from superagent.ab_run import Executor
        ledger=TaskBudgetLedger(args.ledger);before=ledger.project_summary()
        with ledger._db() as db:
            unknowns=[dict(r) for r in db.execute("SELECT * FROM requests WHERE status IN ('unknown','reserved') ORDER BY id")]
            origin=db.execute("SELECT value FROM metadata WHERE key='origin_sha256'").fetchone()[0]
            used=db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(args.scope,)).fetchone()[0]
        if not before['requests'] or origin=='explicit_new_project': raise ValueError('Use the existing nonempty imported team ledger')
        if used: raise ValueError('SA5 scope already contains requests; no automatic rerun')
        for row in registry:
            if sha(args.cases/row['package'])!=row['package_sha256'] or sha(args.cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:
                raise ValueError('Frozen package/oracle digest changed')
        accepted=load_snapshot(args.accepted_history,sha(args.accepted_history)) if args.accepted_history else None
        if accepted and accepted.get('allowed_scope') not in (None,args.scope):
            raise ValueError('Accepted-history authorization is limited to a different experiment scope')
        if accepted and accepted.get('registry_sha256') not in (None,sha(args.cases/'registry.json')):
            raise ValueError('Authorized cases differ from the frozen registry')
        prior={r['id']:r['row_sha256'] for r in accepted['accepted_unknown_requests']} if accepted else {}
        save(args.output/'history_preflight.json',dict(project_before=before,
            already_accepted_unknown_count=len(prior),noncovered_unknowns=[dict(id=r['id'],scope=r['scope'],status=r['status'],
                row_sha256=row_digest(r),course_cost_rmb=None) for r in unknowns if prior.get(r['id'])!=row_digest(r)],
            all_unknown_costs_preserved=True,ledger_modified_by_preflight=False))
        context=accepted_history(args.accepted_history,sha(args.accepted_history)) if accepted else nullcontext()
        with context:
            research_preflight(ledger,args.scope,.80)
            if args.validate_only:
                status='validated_no_requests'
            else:
                if not args.private_output: raise ValueError('Live raw logs need a private output directory')
                args.private_output.mkdir(parents=True,exist_ok=False)
                parent=ExperimentLLM(ENDPOINT,args.key_file,MODEL,.80,thinking_mode='disabled',max_requests=8,
                    allowed_served_models={MODEL},account=IncrementalResearchAccount(ledger,args.scope,str(args.private_output)),streaming=True)
                status='completed'
                with (campus_resolution(CAMPUS_IP),patch.object(Executor,'worker',side_effect=AssertionError('SA5 forbids fitting'))):
                    for case_index,row in enumerate(registry):
                        arms=['legacy','global'] if case_index%2==0 else ['global','legacy']
                        for arm in arms:
                            private=args.private_output/(row['id']+'_'+arm);private.mkdir()
                            logger=RunLogger(private);client=CellClient(parent,arm);clock=time.monotonic()
                            record=dict(case=row['id'],arm=arm)
                            try:
                                path=args.cases/row['package']
                                task=(legacy(path,client,logger) if arm=='legacy' else resolve_training_task(path,client,logger,mode='proactive'))
                                record.update(status='accepted',**grade(task,path,read(args.cases/(row['id']+'.oracle.json'))))
                            except Exception as error:
                                record.update(status='rejected',error_type=type(error).__name__,reason=str(error)[:500])
                            record.update(real_API_requests=client.request_count,elapsed_seconds=time.monotonic()-clock)
                            cells.append(record);save(args.output/'cells.json',cells)
                            if parent.blocked:
                                status='stopped_new_unresolved';break
                        if parent.blocked:break
        if source(root)!=identity: raise RuntimeError('Source changed during panel')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error)[:700])
    after=ledger.project_summary() if ledger else None
    requests=parent.request_count if parent else 0
    receipt=dict(status=status,error=failure,cells=cells,actual_classifier_fits=0,real_API_requests=requests,
        known_incremental_course_cost_rmb=parent.spent if parent else 0,actual_school_debit_rmb=None,
        project_before=before,project_after=after,elapsed_seconds=time.monotonic()-started,
        history_unresolved=True,team_total_verified=False,
        scope='single-turn development semantic probe; no fit, prediction-quality or independent blind-test claim')
    save(args.output/'results.json',receipt);save(args.output/'completion.json',receipt)
    print(json.dumps(dict(status=status,real_API_requests=requests,error=failure),ensure_ascii=False))
    return status not in ('completed','validated_no_requests') or failure is not None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','live'])
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private-output',type=Path);parser.add_argument('--ledger',type=Path)
    parser.add_argument('--accepted-history',type=Path);parser.add_argument('--key-file',type=Path)
    parser.add_argument('--scope',default='sa5_semantic_panel_20261126');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.stage=='prepare': return prepare(args,root)
    if args.ledger is None: parser.error('--ledger is required for live/preflight')
    if not args.validate_only and args.key_file is None: parser.error('--key-file is required for live')
    return live(args,root)


if __name__=='__main__': raise SystemExit(main())
