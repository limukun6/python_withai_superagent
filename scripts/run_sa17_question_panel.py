"""Prepare and run three one-request, controlled real Flash question contracts; zero fits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa12_live_entry import ledger_rows


def sources(root):
    files=list((root/'superagent').glob('*.py'))+[Path(__file__),root/'scripts/sa17_question_cases.py']
    return {p.relative_to(root).as_posix():sha(p) for p in files}


class ControlledClient:
    max_requests=1
    def __init__(self,parent,condition):
        self.parent,self.condition=parent,condition;self.start=parent.request_count;self.initial=parent.spent;self.reply=None
    @property
    def request_count(self):return self.parent.request_count-self.start
    @property
    def blocked(self):return self.parent.blocked
    def ask(self,payload,log,role):
        if self.request_count>=1:raise RuntimeError('Controlled cell cannot spend another request')
        value=deepcopy(payload);value['controlled_interface_condition']=deepcopy(self.condition)
        self.parent.set_deadline(time.monotonic()+120)
        self.reply=self.parent.ask(value,log,role,cost_ceiling=min(self.parent.budget,self.initial+.10))
        return self.reply


def validate(cases):
    registry=read(cases/'registry.json')
    if [r['id'] for r in registry]!=['U1','U2','U3']:raise ValueError('Expected three preregistered controlled cases')
    for row in registry:
        for field,hashed in [('package','package_sha256'),('query','query_sha256')]:
            if sha(cases/row[field])!=row[hashed]:raise ValueError('Controlled input changed')
        if sha(cases/(row['id']+'.condition.json'))!=row['condition_sha256']:raise ValueError('Controlled condition changed')
    return registry


def run(a,root):
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    from superagent.ab_flash import ENDPOINT,MODEL
    from superagent.ab_run import Executor
    from superagent.events import RunLogger
    from superagent.experiment_llm import ExperimentLLM
    from superagent.format_sources import Package,TASK_DOCUMENTS
    from superagent.semantic_contract import resolve_training_task
    from superagent.task_budget import TaskBudgetLedger
    registry=validate(a.cases)
    checked=read(a.checks/'completion.json')
    if checked['status']!='passed' or checked['tests_run']!=292:raise ValueError('Required scoped-question checks have not passed')
    for name,value in read(a.checks/'protocol.json')['source_sha256'].items():
        if sha(root/name)!=value:raise ValueError('Checked implementation changed')
    if read(a.preparation/'completion.json')['status']!='prepared':raise ValueError('Panel not prepared')
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger)
    if before['unresolved_requests']!=16 or before['pending_requests']:raise ValueError('New unknown/pending usage blocks this batch')
    with ledger._db() as db:
        if db.execute('SELECT 1 FROM requests WHERE scope=?',(a.scope,)).fetchone():raise ValueError('No retry of an attempted scope')
    a.private_output.mkdir(parents=True,exist_ok=False)
    record=load_snapshot(a.accepted_history,sha(a.accepted_history))
    record.update(allowed_scope=a.scope,maximum_new_requests=3,estimated_stop_threshold_rmb=.30,
                  parent_authorization_sha256=sha(a.accepted_history),registry_sha256=sha(a.cases/'registry.json'),
                  user_authorization='Existing sixteen unknown rows accepted by user; automatic next steps, registered SA17 controls')
    profile=a.private_output/'accepted_history.json';save(profile,record)
    account=AcceptedHistoryAccount(ledger,a.scope,str(a.private_output),profile,sha(profile));account.preflight()
    if a.validate_only:
        return dict(status='validated_no_requests',ledger_before=before,ledger_after=ledger.project_summary(),
                    actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,historical_rows_unchanged=old==ledger_rows(ledger))
    parent=ExperimentLLM(ENDPOINT,a.key_file,MODEL,.30,max_requests=3,allowed_served_models={MODEL},account=account,
                         streaming=True,campus_ip='10.121.10.250')
    clock=time.monotonic();results=[]
    with patch.object(Executor,'worker',side_effect=AssertionError('Question panel forbids training')):
        for row in registry:
            name=row['id'];control=read(a.cases/(name+'.condition.json'));condition=control['condition'];expected=control['expected']
            client=ControlledClient(parent,condition);run=a.private_output/name;run.mkdir();log=RunLogger(run);log.initialize_usage_ledger()
            data_reads=[];original=Package.read;accepted=False;task=None;error=None;started=time.monotonic()
            def observe(files,member,*args,**kwargs):
                path=Path(member)
                if path.suffix.casefold() not in ('.md','.rst') and path.name.casefold() not in TASK_DOCUMENTS:data_reads.append(member)
                return original(files,member,*args,**kwargs)
            try:
                with patch.object(Package,'read',observe):
                    task=resolve_training_task(a.cases/row['package'],client,log,mode='proactive',prediction_package=a.cases/row['query'])
                accepted=True
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc)[:600])
            events=[json.loads(s) for s in log.events_path.read_text(encoding='utf-8').splitlines()] if log.events_path.exists() else []
            diagnostics=[e['diagnostic'] for e in events if e['kind']=='semantic_contract_rejected']
            reply=client.reply or {};notes=reply.get('uncertainties')
            activated=isinstance(notes,list) and any(isinstance(n,dict) and set(n)=={'field','question','evidence_refs'}
                and n.get('field')==expected['field'] and n.get('evidence_refs')==['doc:README.md:0'] for n in notes)
            activated=activated and reply.get('version') in ('semantic-native-v2','semantic-task-v3')
            if expected['field']=='positive_label':activated=activated and isinstance(reply.get('plan'),dict) and reply['plan'].get('positive_label') is None
            resolution=task.spec.get('uncertainty_resolution') if task is not None else None
            if expected['accepted']:
                mechanism=accepted and resolution is not None and any(
                    n['field']==expected['field'] and n['reason']==expected['resolution_reason'] and n['facts']['class_count']==expected['class_count']
                    for n in resolution['items'])
            else:
                mechanism=not accepted and any(expected['field'] in d.get('uncertainty_fields',[]) for d in diagnostics)
                if expected.get('class_count'):
                    mechanism=mechanism and any(d.get('uncertainty_dependency',{}).get('class_count')==expected['class_count']
                        and d.get('uncertainty_dependency',{}).get('metric')==expected['metric'] for d in diagnostics)
                if expected.get('no_data_members_read'):mechanism=mechanism and not data_reads
            result=dict(case=name,status='passed' if activated and mechanism else 'not_exercised_or_failed',
                controlled_question_activated=activated,expected_acceptance=expected['accepted'],actual_acceptance=accepted,
                executor_mechanism_matches_expectation=bool(mechanism),resolution=resolution,diagnostics=diagnostics,error=error,
                non_document_members_read=data_reads,real_API_requests=client.request_count,known_course_cost_rmb=parent.spent-client.initial,
                actual_classifier_fits=0,elapsed_seconds=time.monotonic()-started)
            results.append(result);evidence=a.output/name;evidence.mkdir();save(evidence/'results.json',result)
            save(evidence/'model_reply.json',client.reply)
            usage=[json.loads(s) for s in log.usage_path.read_text(encoding='utf-8').splitlines()]
            save(evidence/'usage.json',[r for r in usage if r.get('event')!='ledger_initialized'])
            save(a.output/'partial_results.json',results);print(json.dumps(result,ensure_ascii=False),flush=True)
            if parent.blocked or ledger.project_summary()['unresolved_requests']!=16 or ledger.project_summary()['pending_requests']:break
    after=ledger.project_summary();current=ledger_rows(ledger);unchanged=all(current.get(k)==v for k,v in old.items())
    with ledger._db() as db:delta=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(a.scope,))]
    save(a.output/'ledger_delta.json',delta)
    return dict(status='passed' if len(results)==3 and all(r['status']=='passed' for r in results) and unchanged else 'completed_with_failures',
        cases=results,actual_classifier_fits=0,real_API_requests=parent.request_count,known_course_cost_rmb=parent.spent,
        ledger_before=before,ledger_after=after,historical_rows_unchanged=unchanged,elapsed_seconds=time.monotonic()-clock,
        course_budget_compliance='unverified_history',scope='controlled field activation, not natural unknown-task or accuracy/cost-saving evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','live'],required=True)
    for name in ['cases','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source','ledger','key-file','accepted-history','private-output']:p.add_argument('--'+name,type=Path)
    p.add_argument('--scope',default='sa17_scoped_panel_20261127');p.add_argument('--validate-only',action='store_true')
    p.add_argument('--checks',type=Path,default=Path('research/sa17_question_checks_02'))
    p.add_argument('--preparation',type=Path,default=Path('research/sa17_question_prepare_01'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];source=sources(root)
    save(a.output/'protocol.json',dict(batch='SA17-'+a.mode,base='1ecc4c9',source_sha256=source,
        plan_sha256=sha(root/'docs/batches/SA17_scoped_questions_plan_20261127_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0 if a.mode=='prepare' or a.validate_only else 3,
        request_limit_per_cell=1,estimated_stop_threshold_rmb=0 if a.mode=='prepare' else .30,
        controlled_question_activation=True,natural_activation=False,automatic_retry=False))
    try:
        if a.mode=='prepare':
            from scripts.sa17_question_cases import build
            registry=build(a.cases,a.source);save(a.output/'registry.json',registry)
            result=dict(status='prepared',actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,registry_sha256=sha(a.cases/'registry.json'))
        else:result=run(a,root)
    except Exception as exc:
        result=dict(status='failed',error=dict(type=type(exc).__name__,message=str(exc)[:900]),
                    note='Inspect real ledger and partial results; failures are never assumed zero or retried')
    result['source_unchanged']=source==sources(root)
    if not result['source_unchanged']:result['status']='failed'
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
    return result['status'] not in ('passed','prepared','validated_no_requests')


if __name__=='__main__':raise SystemExit(main())
