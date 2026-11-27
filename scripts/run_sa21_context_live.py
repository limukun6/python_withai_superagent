"""SA21: natural production solves on a new public data source with bounded action context."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/SA21_ACTION_CONTEXT_PLAN_20261127_zh.md'


def identity():
    paths=list((ROOT/'superagent').glob('*.py'))+[ROOT/'unified_agent.py',Path(__file__),
        ROOT/'scripts/sa21_context_cases.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def prepare(a):
    from scripts.sa21_context_cases import build
    from superagent.semantic_contract import resolve_training_task
    a.output.mkdir(parents=True,exist_ok=False)
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Checks must pass')
    frozen=read(a.checks/'protocol.json')['source_sha256']
    if any(sha(ROOT/n)!=v for n,v in frozen.items()):raise ValueError('Checked source changed')
    registry=build(a.output/'cases');local=[]
    for row in registry:
        try:
            task=resolve_training_task(a.output/'cases'/row['package'],prediction_package=a.output/'cases'/row['query'])
            local.append(dict(id=row['id'],status='readable',kind=task.kind))
        except Exception as error:local.append(dict(id=row['id'],status='input_failed',error_type=type(error).__name__))
    save(a.output/'local_intake.json',dict(cases=local,actual_classifier_fits=0,real_API_requests=0))
    protocol=dict(batch='SA21-new-source-context',base='0b0f5c9',source_sha256=identity(),plan_sha256=sha(PLAN),
        registry_sha256=sha(a.output/'cases/registry.json'),checks_completion_sha256=sha(a.checks/'completion.json'),
        recoverable_cases=['C1','C2'],unresolved_control=[],maximum_fits_per_path=4,maximum_API_per_path=3,
        maximum_total_fits=8,maximum_total_API=6,estimated_stop_threshold_rmb_per_path=.30,wall_seconds_per_path=360,
        fault_scope='none; natural model responses via unmodified production solve',
        oracle_access='grading only after execution; not in model workspace',
        evidence_boundary='one new Breast Cancer source, two controlled packaging conditions; not two independent tasks, not a quality or causal repair comparison')
    save(a.output/'protocol.json',protocol)
    save(a.output/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_API_requests=0))
    print(dict(status='prepared',local=local))


def clean_env():
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
    env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def execute(command,env,log,timeout=400):
    with log.open('w',encoding='utf-8') as stream:
        return subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,timeout=timeout).returncode


def run(a):
    from superagent.task_budget import TaskBudgetLedger
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    from superagent.semantic_artifact import load_artifact
    from scripts.run_sa14_native_delivery import grade
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();protocol=read(a.prepared/'protocol.json')
    if protocol['source_sha256']!=identity() or protocol['plan_sha256']!=sha(PLAN):raise ValueError('Frozen source or plan changed')
    cases=a.prepared/'cases'
    if sha(cases/'registry.json')!=protocol['registry_sha256']:raise ValueError('Registry changed')
    registry=read(cases/'registry.json')
    for row in registry:
        for key in ('package','query'):
            if sha(cases/row[key])!=row[key+'_sha256']:raise ValueError('Input changed')
        if sha(cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Oracle changed')
    save(a.output/'protocol.json',dict(protocol,prepared_protocol_sha256=sha(a.prepared/'protocol.json'),scope_prefix=a.scope))
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger)
    if before['unresolved_requests']!=16 or before['pending_requests']:raise ValueError('New unresolved usage blocks paid batch')
    prior=load_snapshot(a.accepted_history,sha(a.accepted_history))
    if len(prior['accepted_unknown_requests'])!=16:raise ValueError('Expected exactly the already authorized sixteen unknowns')
    rows=[];fatal=None
    for row in registry:
        name=row['id'];scope=a.scope+'_'+name.lower();out=a.output/name;private=a.private_output/name
        out.mkdir();private.mkdir();run_dir=private/'run';clock=time.monotonic();result=dict(id=name,recoverable=row['recoverable'])
        try:
            with ledger._db() as db:
                if db.execute('SELECT 1 FROM requests WHERE scope=?',(scope,)).fetchone():raise ValueError('Scope already attempted: no retry')
            profile=dict(prior,user_authorization='User requested autonomous next steps under existing accepted sixteen unknowns; SA21 frozen two-condition new-source context batch',
                parent_authorization_sha256=sha(a.accepted_history),allowed_scope=scope,maximum_new_requests=3,
                estimated_stop_threshold_rmb=.30,registry_sha256=protocol['registry_sha256'])
            profile_path=private/'accepted_history.json';save(profile_path,profile)
            AcceptedHistoryAccount(ledger,scope,str(run_dir),profile_path,sha(profile_path)).preflight()
            env=clean_env();env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
                SUPERAGENT_TASK_BUDGET_ID=scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(profile_path.resolve()),
                SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(profile_path),SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',SUPERAGENT_OFFLINE='0')
            package=str((cases/row['package']).resolve());query=str((cases/row['query']).resolve())
            destination=run_dir/'delivery/predictions.csv'
            command=[sys.executable,'-m','unified_agent','solve',package,'--prediction-input',query,
                '--fit-limit','4','--time-limit','360','--run-dir',str(run_dir.resolve()),'--output',str(destination.resolve())]
            save(out/'invocation.json',dict(command=command,scope=scope,accepted_history_sha256=sha(profile_path),
                package_sha256=row['package_sha256'],query_sha256=row['query_sha256'],oracle_sent=False))
            code=execute(command,env,private/'execute.log');result['returncode']=code
            agent=run_dir/'agent';state=read(agent/'recovery_state.json')
            save(out/'recovery_state.json',state)
            result['premature_unresolved']=bool(row['recoverable'] and state['status']=='unresolved')
            if code:raise RuntimeError('Formal entry did not finish; see private execute.log')
            selection=read(agent/'final_selection.json')
            if not selection.get('refit') or selection.get('delivery_status')=='partial':raise ValueError('Only a partial checkpoint was delivered')
            result['false_completion']=True  # Clear only after the independent semantic/output checks.
            task=load_artifact(agent/'semantic_artifact.json',package,query)
            oracle=read(cases/(name+'.oracle.json'));verification=grade(task,query,oracle['expected'])
            save(out/'semantic_grade.json',verification)
            if not verification['complete']:raise ValueError('Frozen semantic oracle differs from accepted task')
            prediction_command=[sys.executable,'-m','unified_agent','predict',query,'--run-dir',str(run_dir.resolve()),'--output',str(destination.resolve())]
            replay=private/'replayed.csv';prediction_command[-1]=str(replay.resolve())
            if execute(prediction_command,clean_env(),private/'replay.log',60):raise RuntimeError('Fresh-process replay failed')
            if sha(replay)!=sha(destination):raise ValueError('Fresh-process predictions differ')
            recipe=oracle['expected']['output']
            with destination.open(encoding=recipe['encoding'],newline='') as stream:
                records=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
            if len(records)!=row['prediction_rows'] or list(records[0])!=[c['name'] for c in recipe['columns']]:raise ValueError('Wrong output dimensions')
            if recipe['encoding']=='utf-8-sig' and not destination.read_bytes().startswith(b'\xef\xbb\xbf'):raise ValueError('Required BOM absent')
            for column in recipe['columns']:
                values=[r[column['name']] for r in records]
                if column['kind']=='input_id' and values!=oracle['prediction_ids']:raise ValueError('Original IDs changed')
                if column['kind']=='row_index' and values!=[str(i+column['start']) for i in range(len(records))]:raise ValueError('Wrong row order')
                if column['kind']=='label' and not set(values)<=set(map(str,oracle['expected']['classes'])):raise ValueError('Wrong labels')
                if column['kind']=='probability' and not all(0<=float(v)<=1 for v in values):raise ValueError('Invalid probability')
            failures=state['failed_actions']
            if not failures:raise ValueError('No recorded failed operation before recovery')
            result.update(status='passed',complete_delivery=True,false_completion=False,
                semantic_checks=verification['checks'],failure_records=len(failures),structure_observations=len(state['observed_evidence']),
                prediction_rows=len(records),fresh_process_equal=True,model_sha256=selection['model_sha256'],
                prediction_sha256=sha(destination),winner=selection['family'])
            save(out/'final_selection.json',selection);save(out/'semantic_contract.json',read(agent/'semantic_contract.json'))
        except Exception as e:
            result.update(status='failed',complete_delivery=False,error=dict(type=type(e).__name__,message=str(e)[:700]))
        agent=run_dir/'agent'
        fit=read(agent/'classifier_fits.json') if (agent/'classifier_fits.json').is_file() else {'attempted_classifier_fits':0}
        save(out/'classifier_fits.json',fit);result['actual_classifier_fits']=fit['attempted_classifier_fits']
        usage=agent/'llm_usage.jsonl'
        calls=[json.loads(s) for s in usage.read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')=='llm_call'] if usage.is_file() else []
        save(out/'metered_usage.json',calls)
        result.update(intake_real_API_requests=sum(u['role'].startswith('task_semantic') for u in calls),
                      planner_real_API_requests=sum(u['role']=='experiment_planner' for u in calls))
        report=agent/'unresolved_task_report.json'
        if report.is_file():save(out/'unresolved_task_report.json',read(report))
        result['explicit_unresolved_report']=report.is_file()
        events_file=agent/'events.jsonl'
        if events_file.is_file():
            events=[json.loads(s) for s in events_file.read_text(encoding='utf-8').splitlines()]
            result['batch_inspection_requests']=sum(e['kind']=='llm_response' and '"inspect_batch"' in e.get('content','') for e in events)
            packets=[json.loads(e['messages'][-1]['content']) for e in events if e['kind']=='llm_request']
            result['requests_with_failed_action_summary']=sum(any(r.get('action_summary') is not None for r in packet.get('recovery_state',{}).get('failed_actions',[])) for packet in packets)
            result['natural_model_contract_failures']=sum(e['kind']=='semantic_contract_rejected' and isinstance(e.get('proposal'),dict) and 'version' in e['proposal'] for e in events)


        with ledger._db() as db:delta=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(scope,))]
        save(out/'ledger_delta.json',delta)
        result.update(real_API_requests=len(delta),known_course_cost_rmb=sum(r['course_units'] or 0 for r in delta)/1e9,
            elapsed_seconds=time.monotonic()-clock,human_interventions_during_execution=0)
        after=ledger.project_summary()
        if after['unresolved_requests']!=16 or after['pending_requests']:
            fatal='new_unresolved_usage';result['status']='stopped_new_unresolved'
        if result['actual_classifier_fits']>4 or len(delta)>3:fatal='resource_bound_exceeded'
        save(out/'completion.json',result);rows.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if fatal:break
    current=ledger_rows(ledger);unchanged=all(current.get(k)==v for k,v in old.items())
    all_passed=len(rows)==2 and all(r['status'] in ('passed','unresolved_control_retained') for r in rows)
    result=dict(status='stopped' if fatal else 'passed' if all_passed else 'failed',cases=rows,
        recoverable_correct_deliveries=sum(r.get('complete_delivery',False) for r in rows if r['recoverable']),recoverable_planned=2,
        false_completions=sum(r.get('false_completion',False) for r in rows),
        actual_classifier_fits=sum(r['actual_classifier_fits'] for r in rows),real_API_requests=sum(r['real_API_requests'] for r in rows),
        known_course_cost_rmb=sum(r['known_course_cost_rmb'] for r in rows),elapsed_seconds=time.monotonic()-start,
        historical_rows_unchanged=unchanged,source_unchanged=protocol['source_sha256']==identity(),
        ledger_before=before,ledger_after=ledger.project_summary(),fatal=fatal,course_budget_compliance='unverified_history')
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',type=Path,required=True);q.add_argument('--checks',type=Path,required=True)
    q=sub.add_parser('run')
    for key in ('prepared','output','private-output','ledger','key-file','accepted-history'):q.add_argument('--'+key,type=Path,required=True)
    q.add_argument('--scope',default='sa21_context_20261127_01');a=p.parse_args()
    return prepare(a) if a.mode=='prepare' else run(a)


if __name__=='__main__':main()
