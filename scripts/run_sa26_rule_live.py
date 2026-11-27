"""Prepare then run a bounded natural-model rule-recovery panel on one real ledger."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa21_context_live import clean_env

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/SA26_RULE_RECOVERY_PLAN_20261127_zh.md'


def identity():
    paths=list((ROOT/'superagent').glob('*.py'))+[ROOT/'unified_agent.py',Path(__file__),ROOT/'scripts/sa26_rule_cases.py',ROOT/'tools/import_semantic_artifact.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def execute(command,env,log,timeout=410):
    with log.open('w',encoding='utf-8') as stream:
        return subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode


def prepare(a):
    from scripts.sa26_rule_cases import build
    from superagent.semantic_contract import resolve_training_task
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Rule checks must pass')
    if any(sha(ROOT/n)!=v for n,v in read(a.checks/'protocol.json')['source_sha256'].items()):raise ValueError('Checked source changed')
    a.output.mkdir(parents=True,exist_ok=False);registry=build(a.output/'cases');local=[]
    for row in registry:
        try:resolve_training_task(a.output/'cases'/row['package']);state='accepted';error=None
        except Exception as issue:state='input_failed';error=dict(type=type(issue).__name__,message=str(issue)[:700])
        local.append(dict(id=row['id'],status=state,error=error))
    save(a.output/'local_baseline.json',dict(cases=local,actual_classifier_fits=0,campus_API_requests=0))
    save(a.output/'protocol.json',dict(batch='SA26-rule-recovery',base='91f1a8c',source_sha256=identity(),plan_sha256=sha(PLAN),
        registry_sha256=sha(a.output/'cases/registry.json'),checks_completion_sha256=sha(a.checks/'completion.json'),
        per_positive_fit_limit=2,per_path_max_API=3,per_path_estimated_stop_threshold_rmb=.30,maximum_total_fits=4,maximum_total_API=9,
        positive_cases=['R1','R2'],negative_cases=['N1'],order=[r['id'] for r in registry],wall_seconds_per_path=360,
        oracle_access='grading only, never in task package or messages',reply_injection=False,manual_model_selection=False,
        scope='controlled natural-language rule conditions on previously used sources, not unseen task generality or quality'))
    save(a.output/'completion.json',dict(status='prepared',actual_classifier_fits=0,campus_API_requests=0))
    print(dict(status='prepared',local_baseline=local))


def run(a):
    from superagent.task_budget import TaskBudgetLedger
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    from superagent.semantic_artifact import load_artifact
    from superagent.metrics import MetricSpec
    from superagent.data import load_prediction_features
    from scripts.sa7_multiturn_cases import array_fingerprint
    protocol=read(a.prepared/'protocol.json');cases=a.prepared/'cases'
    if protocol['source_sha256']!=identity() or protocol['plan_sha256']!=sha(PLAN):raise ValueError('Source or plan changed')
    if sha(cases/'registry.json')!=protocol['registry_sha256']:raise ValueError('Registry changed')
    registry=read(cases/'registry.json')
    for row in registry:
        for key in ('package','query'):
            if sha(cases/row[key])!=row[key+'_sha256']:raise ValueError('Input changed')
        if sha(cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Oracle changed')
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(protocol,scope_prefix=a.scope))
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger);clock=time.monotonic()
    prior=load_snapshot(a.accepted_history,sha(a.accepted_history))
    expected_unknowns=len(prior['accepted_unknown_requests'])
    if before['pending_requests'] or before['unresolved_requests']!=expected_unknowns:raise ValueError('Unexpected ledger state for explicitly enumerated snapshot')
    results=[];fatal=None
    save(a.output/'historical_rows_before.json',old)
    for row in registry:
        name=row['id'];scope=a.scope+'_'+name.lower();out=a.output/name;private=a.private_output/name;out.mkdir();private.mkdir()
        run_dir=private/'run';model=run_dir/'agent';start=time.monotonic()
        result=dict(id=name,recoverable=row['recoverable'],status='failed',complete_delivery=False,false_completion=False)
        try:
            with ledger._db() as db:
                if db.execute('SELECT 1 FROM requests WHERE scope=?',(scope,)).fetchone():raise ValueError('Used scope cannot be retried')
            profile=dict(prior,user_authorization='2026-11-27 user authorized continuing bounded paid research with current unknown requests preserved; SA26 fixed three rule cases',
                parent_authorization_sha256=sha(a.accepted_history),allowed_scope=scope,maximum_new_requests=3,
                estimated_stop_threshold_rmb=.30,registry_sha256=protocol['registry_sha256'])
            path=private/'accepted_history.json';save(path,profile);AcceptedHistoryAccount(ledger,scope,str(run_dir),path,sha(path)).preflight()
            env=clean_env();env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
                SUPERAGENT_TASK_BUDGET_ID=scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(path.resolve()),SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(path),
                SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',SUPERAGENT_OFFLINE='0')
            package=(cases/row['package']).resolve();query=(cases/row['query']).resolve();destination=private/'delivery/predictions.csv'
            args=[sys.executable,'-m','unified_agent','solve',str(package),'--prediction-input',str(query),'--fit-limit','2','--time-limit','360',
                  '--run-dir',str(run_dir.resolve()),'--output',str(destination.resolve())]
            save(out/'invocation.json',dict(command=args,scope=scope,accepted_history_sha256=sha(path),stdin='closed',injected_reply=False,oracle_sent=False))
            code=execute(args,env,private/'execute.log');result['returncode']=code
            outcome=read(model/'run_outcome.json') if (model/'run_outcome.json').is_file() else {}
            save(out/'run_outcome.json',outcome)
            state=read(model/'recovery_state.json');save(out/'recovery_state.json',state)
            if row['recoverable']:
                if code or outcome.get('status')!='completed':raise ValueError('Recoverable task did not complete')
                selection=read(model/'final_selection.json');task=load_artifact(model/'semantic_artifact.json',package,query)
                oracle=read(cases/(name+'.oracle.json'));expected=oracle['expected'];metric=MetricSpec.for_task(task)
                values=load_prediction_features(query,task.kind,task.spec);rules=task.spec.get('rule_resolution',{}).get('rules',[])
                checks=dict(kind=task.kind==expected['kind'],metric=metric.name==expected['metric'],classes=list(metric.labels)==expected['classes'],
                    training_features=array_fingerprint(task.X_train)==expected['training_features_sha256'],training_labels=array_fingerprint(task.y_train)==expected['training_labels_sha256'],
                    prediction_features=array_fingerprint(values)==expected['prediction_features_sha256'],restrictions=task.spec.get('method_restrictions',[])==expected['required_restrictions'],
                    rule_record=any(r['field']==expected['normalized_field'] and r['value']==expected['normalized_value'] for r in rules),
                    full_refit=selection['refit'] is True)
                save(out/'semantic_grade.json',checks)
                if not all(checks.values()):result['false_completion']=True;raise ValueError('Completed task violates frozen oracle')
                with destination.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
                if len(rows)!=24 or list(rows[0])!=['Row','Class'] or [r['Row'] for r in rows]!=[str(i+1) for i in range(24)] or not {r['Class'] for r in rows}<=set(expected['classes']):
                    result['false_completion']=True;raise ValueError('Wrong output meaning/order/domain')
                replay=private/'replay/predictions.csv';cmd=[sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(run_dir.resolve()),'--output',str(replay.resolve())]
                if execute(cmd,clean_env(),private/'replay.log',60) or sha(replay)!=sha(destination):raise ValueError('Independent process replay differs')
                save(out/'semantic_contract.json',read(model/'semantic_contract.json'));save(out/'task_spec.json',read(model/'task_spec.json'));save(out/'final_selection.json',selection)
                result.update(status='passed',complete_delivery=True,false_completion=False,rows=len(rows),rules=rules,fresh_process_equal=True,
                    prediction_sha256=sha(destination),model_sha256=selection['model_sha256'])
            else:
                fits=read(model/'classifier_fits.json')['attempted_classifier_fits'] if (model/'classifier_fits.json').is_file() else 0
                if outcome.get('status')=='completed' or fits:result['false_completion']=True;raise ValueError('Contradiction was trained or declared complete')
                if not state['failed_actions']:raise ValueError('No recovery evidence for contradiction')
                result.update(status='unresolved_control_retained',complete_delivery=False,actual_classifier_fits=0,
                    explicit_unresolved_report=(model/'unresolved_task_report.json').is_file(),outcome_status=outcome.get('status'))
            if (model/'unresolved_task_report.json').is_file():save(out/'unresolved_task_report.json',read(model/'unresolved_task_report.json'))
        except Exception as issue:result['error']=dict(type=type(issue).__name__,message=str(issue)[:800])
        fit=model/'classifier_fits.json';account=read(fit) if fit.is_file() else dict(attempted_classifier_fits=0)
        save(out/'classifier_fits.json',account);result['actual_classifier_fits']=account['attempted_classifier_fits']
        usage=model/'llm_usage.jsonl';calls=[json.loads(s) for s in usage.read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')=='llm_call'] if usage.is_file() else []
        save(out/'metered_usage.json',calls)
        events=model/'events.jsonl'
        if events.is_file():save(out/'semantic_events.json',[json.loads(s) for s in events.read_text(encoding='utf-8').splitlines() if json.loads(s)['kind'] in ('llm_response','semantic_contract_rejected','semantic_contract_accepted')])
        with ledger._db() as db:delta=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(scope,))]
        save(out/'ledger_delta.json',delta);after=ledger.project_summary()
        result.update(real_API_requests=len(delta),known_course_cost_rmb=sum(r['course_units'] or 0 for r in delta)/1e9,
            elapsed_seconds=time.monotonic()-start,human_interventions_during_execution=0)
        if after['pending_requests'] or after['unresolved_requests']!=expected_unknowns or any(r['status']!='known' for r in delta):fatal='new_unresolved_usage'
        if len(delta)>3 or result['actual_classifier_fits']>(2 if row['recoverable'] else 0):fatal='resource_limit_exceeded'
        save(out/'completion.json',result);results.append(result);save(a.output/'partial_results.json',results);print(json.dumps(result),flush=True)
        if fatal:break
    current=ledger_rows(ledger);unchanged=all(current.get(k)==v for k,v in old.items())
    record=dict(status='stopped' if fatal else 'passed' if len(results)==3 and all(r['status'] in ('passed','unresolved_control_retained') for r in results) else 'failed',
        cases=results,actual_classifier_fits=sum(r['actual_classifier_fits'] for r in results),real_API_requests=sum(r['real_API_requests'] for r in results),
        known_course_cost_rmb=sum(r['known_course_cost_rmb'] for r in results),elapsed_seconds=time.monotonic()-clock,
        historical_rows_unchanged=unchanged,source_unchanged=protocol['source_sha256']==identity(),ledger_before=before,ledger_after=ledger.project_summary(),fatal=fatal,
        quality_evaluation=False,course_budget_compliance='unverified_history')
    save(a.output/'results.json',record);save(a.output/'completion.json',record);print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare')
    for key in ('output','checks'):q.add_argument('--'+key,type=Path,required=True)
    q=sub.add_parser('run')
    for key in ('prepared','output','private-output','ledger','key-file','accepted-history'):q.add_argument('--'+key,type=Path,required=True)
    q.add_argument('--scope',default='sa26_rule_20261127_01');a=p.parse_args()
    return prepare(a) if a.mode=='prepare' else run(a)


if __name__=='__main__':main()
