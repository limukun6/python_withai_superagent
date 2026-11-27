"""SA24: preregister old/new image solves with free and partially fixed preprocessing."""
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
from scripts.run_sa21_context_live import clean_env

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/SA24_IMAGE_DESIGN_PLAN_20261127_zh.md'


def identity(root):
    paths=list((root/'superagent').glob('*.py'))+[root/'unified_agent.py']
    return {p.relative_to(root).as_posix():sha(p) for p in paths}


def harness_identity():
    return {n:sha(ROOT/n) for n in ('scripts/sa24_image_cases.py','scripts/run_sa24_image_live.py')}


def execute(command,env,log,cwd,timeout=410):
    with log.open('w',encoding='utf-8') as stream:
        return subprocess.run(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,timeout=timeout).returncode


def prepare(a):
    from scripts.sa24_image_cases import build
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Required checks failed')
    if any(sha(ROOT/n)!=v for n,v in read(a.checks/'protocol.json')['source_sha256'].items()):raise ValueError('Checked source changed')
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.baseline,text=True).strip()
    if not base.startswith('1887ffa'):raise ValueError('Unexpected baseline')
    if subprocess.check_output(['git','status','--porcelain'],cwd=a.baseline,text=True).strip():raise ValueError('Baseline not clean')
    a.output.mkdir(parents=True,exist_ok=False);registry=build(a.output/'cases')
    save(a.output/'protocol.json',dict(batch='SA24-image-design',baseline_commit=base,baseline_root=str(a.baseline.resolve()),
        baseline_source_sha256=identity(a.baseline),source_sha256=identity(ROOT),harness_sha256=harness_identity(),
        plan_sha256=sha(PLAN),checks_completion_sha256=sha(a.checks/'completion.json'),registry_sha256=sha(a.output/'cases/registry.json'),
        order=[dict(id='F_old',case='F',version='old'),dict(id='F_new',case='F',version='new'),
               dict(id='P_new',case='P',version='new'),dict(id='P_old',case='P',version='old')],
        per_path=dict(actual_fit_limit=2,API_limit=3,wall_seconds=360,estimated_stop_threshold_rmb=.30),
        maximum_total_fits=8,maximum_total_API=12,seed='same production default; no manual algorithm selection',
        oracle_access='grading only after formal solve; no proposal, reply, model, or tool injection',
        evidence_boundary='one previously used Digits source; controlled design conditions, no accuracy or independent-task claims'))
    save(a.output/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_API_requests=0))
    print(dict(status='prepared',cases=[r['id'] for r in registry]))


def run(a):
    from superagent.task_budget import TaskBudgetLedger
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    from scripts.sa24_image_cases import grade
    protocol=read(a.prepared/'protocol.json');baseline=Path(protocol['baseline_root']);cases=a.prepared/'cases'
    def frozen():
        return protocol['source_sha256']==identity(ROOT) and protocol['baseline_source_sha256']==identity(baseline) and protocol['harness_sha256']==harness_identity() and protocol['plan_sha256']==sha(PLAN)
    if not frozen():raise ValueError('Frozen source/plan changed')
    if sha(cases/'registry.json')!=protocol['registry_sha256']:raise ValueError('Registry changed')
    registry={r['id']:r for r in read(cases/'registry.json')}
    for row in registry.values():
        for key in ('package','query'):
            if sha(cases/row[key])!=row[key+'_sha256']:raise ValueError('Input changed')
        if sha(cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Oracle changed')
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(protocol,prepared_protocol_sha256=sha(a.prepared/'protocol.json'),scope_prefix=a.scope))
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger);start=time.monotonic()
    if before['unresolved_requests']!=17 or before['pending_requests']:raise ValueError('Unexpected unresolved ledger state')
    prior=load_snapshot(a.accepted_history,sha(a.accepted_history));results=[];fatal=None
    save(a.output/'historical_rows_before.json',old)
    for cell in protocol['order']:
        name=cell['id'];row=registry[cell['case']];scope=a.scope+'_'+name.lower();cwd=baseline if cell['version']=='old' else ROOT
        out=a.output/name;private=a.private_output/name;out.mkdir();private.mkdir();run_dir=private/'run';clock=time.monotonic()
        result=dict(**cell,status='failed',complete_delivery=False,false_completion=False,formal_exit_success=False)
        try:
            if not frozen():raise ValueError('Source changed during batch')
            with ledger._db() as db:
                if db.execute('SELECT 1 FROM requests WHERE scope=?',(scope,)).fetchone():raise ValueError('Attempted scope cannot be retried')
            profile=dict(prior,user_authorization='2026-11-27 user authorized ignoring unknown usage and continuing bounded paid research; SA24 four frozen old/new image paths',
                parent_authorization_sha256=sha(a.accepted_history),allowed_scope=scope,maximum_new_requests=3,
                estimated_stop_threshold_rmb=.30,registry_sha256=protocol['registry_sha256'])
            path=private/'accepted_history.json';save(path,profile)
            AcceptedHistoryAccount(ledger,scope,str(run_dir),path,sha(path)).preflight()
            env=clean_env();env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
                SUPERAGENT_TASK_BUDGET_ID=scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(path.resolve()),
                SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(path),SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',SUPERAGENT_OFFLINE='0')
            package=(cases/row['package']).resolve();query=(cases/row['query']).resolve();destination=private/'delivery/predictions.csv'
            command=[sys.executable,'-m','unified_agent','solve',str(package),'--prediction-input',str(query),
                '--fit-limit','2','--time-limit','360','--run-dir',str(run_dir.resolve()),'--output',str(destination.resolve())]
            save(out/'invocation.json',dict(command=command,cwd=str(cwd),scope=scope,accepted_history_sha256=sha(path),
                package_sha256=row['package_sha256'],query_sha256=row['query_sha256'],oracle_sent=False,stdin='closed'))
            result['returncode']=execute(command,env,private/'execute.log',cwd)
            if result['returncode']:raise RuntimeError('Formal solve failed; private execute.log retained')
            result['formal_exit_success']=True;agent=run_dir/'agent';selection=read(agent/'final_selection.json')
            if not selection.get('refit') or selection.get('delivery_status')=='partial':raise ValueError('No full final refit')
            proposal=read(agent/'semantic_contract.json')['proposal'];spec=read(agent/'task_spec.json');oracle=read(cases/(row['id']+'.oracle.json'))
            verification=grade(package,query,proposal,spec,oracle);save(out/'semantic_grade.json',verification)
            if not verification['complete']:
                result['false_completion']=True;raise ValueError('Completed output violates frozen task oracle')
            with destination.open(encoding='utf-8',newline='') as stream:predictions=list(csv.DictReader(stream))
            valid=(len(predictions)==30 and list(predictions[0])==['Row','Class'] and
                [p['Row'] for p in predictions]==[str(i+1) for i in range(30)] and
                {p['Class'] for p in predictions}<=set(oracle['classes']))
            if not valid:
                result['false_completion']=True;raise ValueError('CSV violates required header/order/classes')
            protected={str(p):sha(p) for p in [agent/selection['model_file'],agent/'task_spec.json',agent/'final_selection.json',agent/'classifier_fits.json']}
            replay=private/'replay/predictions.csv'
            predict=[sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(run_dir.resolve()),'--output',str(replay.resolve())]
            if execute(predict,clean_env(),private/'replay.log',cwd,60):raise RuntimeError('Independent offline prediction failed')
            if sha(replay)!=sha(destination) or protected!={n:sha(n) for n in protected}:raise ValueError('Independent replay changed export or frozen model')
            manifest=read(destination.parent/'prediction_manifest.json')
            if manifest['delivery_status']!='completed':raise ValueError('Incomplete delivery receipt')
            save(out/'frozen_files.json',protected);save(out/'prediction_manifest.json',manifest)
            save(out/'final_selection.json',selection);save(out/'semantic_contract.json',proposal);save(out/'task_spec.json',spec)
            result.update(status='passed',complete_delivery=True,rows=len(predictions),fresh_process_equal=True,
                chosen=verification['chosen'],provenance_correct=verification['provenance_correct'],
                proposal_version=proposal['version'],family=selection['family'],model_sha256=selection['model_sha256'],
                prediction_sha256=sha(destination),replay_new_API=0)
        except Exception as issue:result['error']=dict(type=type(issue).__name__,message=str(issue)[:700])
        agent=run_dir/'agent';fit=agent/'classifier_fits.json'
        account=read(fit) if fit.is_file() else dict(attempted_classifier_fits=0)
        save(out/'classifier_fits.json',account);result['actual_classifier_fits']=account['attempted_classifier_fits']
        usage=agent/'llm_usage.jsonl';events=agent/'events.jsonl'
        calls=[json.loads(s) for s in usage.read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')=='llm_call'] if usage.is_file() else []
        save(out/'metered_usage.json',calls)
        if events.is_file():
            items=[json.loads(s) for s in events.read_text(encoding='utf-8').splitlines()]
            save(out/'semantic_events.json',[e for e in items if e['kind'] in ('llm_response','semantic_contract_accepted','semantic_contract_rejected','semantic_structure_observed')])
        for filename in ('recovery_state.json','unresolved_task_report.json'):
            if (agent/filename).is_file():save(out/filename,read(agent/filename))
        with ledger._db() as db:delta=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(scope,))]
        save(out/'ledger_delta.json',delta)
        result.update(real_API_requests=len(delta),known_course_cost_rmb=sum(r['course_units'] or 0 for r in delta)/1e9,
            intake_API=sum(c['role'].startswith('task_semantic') for c in calls),planner_API=sum(c['role']=='experiment_planner' for c in calls),
            elapsed_seconds=time.monotonic()-clock,human_interventions_during_execution=0)
        after=ledger.project_summary()
        if after['unresolved_requests']!=17 or after['pending_requests']:fatal='new_unresolved_usage'
        if any(r['status']!='known' for r in delta):fatal='request_failure_no_retry'
        if result['actual_classifier_fits']>2 or len(delta)>3:fatal='resource_limit_exceeded'
        save(out/'completion.json',result);results.append(result);save(a.output/'partial_results.json',results)
        print(json.dumps(result),flush=True)
        if fatal:break
    current=ledger_rows(ledger);unchanged=all(current.get(k)==v for k,v in old.items())
    result=dict(status='stopped' if fatal else 'passed' if len(results)==4 and all(r['status']=='passed' for r in results) else 'completed_with_failures',
        cases=results,actual_classifier_fits=sum(r['actual_classifier_fits'] for r in results),real_API_requests=sum(r['real_API_requests'] for r in results),
        known_course_cost_rmb=sum(r['known_course_cost_rmb'] for r in results),elapsed_seconds=time.monotonic()-start,
        ledger_before=before,ledger_after=ledger.project_summary(),historical_rows_unchanged=unchanged,source_unchanged=frozen(),fatal=fatal,
        quality_evaluation=False,course_budget_compliance='unverified_history')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare')
    for key in ('output','checks','baseline'):p.add_argument('--'+key,type=Path,required=True)
    p=sub.add_parser('run')
    for key in ('prepared','output','private-output','ledger','key-file','accepted-history'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scope',default='sa24_image_20261127_01');args=parser.parse_args()
    return prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':main()
