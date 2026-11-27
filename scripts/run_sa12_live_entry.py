"""One registered fresh, real Flash solve via production child processes; no injected clients."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read, save, sha


def ledger_rows(ledger):
    from superagent.accepted_history import row_digest
    with ledger._db() as db:
        return {r['id']: row_digest(r) for r in db.execute('SELECT * FROM requests')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','ledger','key-file','accepted-history','output','private-output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',default='sa12_live_entry_20261126')
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];clock=time.monotonic()
    identity={f.relative_to(root).as_posix():sha(f) for f in (root/'superagent').glob('*.py')}
    identity.update({'unified_agent.py':sha(root/'unified_agent.py'), 'scripts/run_sa12_live_entry.py':sha(Path(__file__))})
    save(a.output/'protocol.json',dict(batch='SA12-fresh-live-production-entry',base='2304403',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA12_live_entry_plan_20261126_zh.md'),validate_only=a.validate_only,
        maximum_classifier_fits=2,maximum_real_API_requests=3,estimated_stop_threshold_rmb=.25,wall_seconds=240,
        case='V1 from previously used SA8 real development package; not a new transfer sample',
        intervention='none: standalone production solve, stdin closed; no oracle gate or injected clients/resolvers/classifiers',
        quality_evaluation=False,authorization_parent_sha256=sha(a.accepted_history)))
    ledger=None;before=None;old_rows={};error=None;status='blocked_preflight';run=a.private_output/'V1';used=0
    delivery=None;returncode=None
    try:
        from superagent.task_budget import TaskBudgetLedger
        from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
        checks=read(root/'research/sa12_live_entry_checks_02/completion.json')
        if checks['status']!='passed' or checks['tests_run']!=223:
            raise ValueError('Required SA12 no-fit/no-API checks are not passed')
        if any(checks_key not in checks for checks_key in ['source_unchanged']):raise ValueError('Missing source audit')
        frozen_checks=read(root/'research/sa12_live_entry_checks_02/protocol.json')['source_sha256']
        if any(sha(root/name)!=digest for name,digest in frozen_checks.items()):raise ValueError('Checked implementation changed')
        row=next(r for r in read(a.cases/'registry.json') if r['id']=='V1')
        for field,hashfield in [('package','package_sha256'),('query','query_sha256')]:
            if sha(a.cases/row[field])!=row[hashfield]:raise ValueError('Registered task package changed')
        ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old_rows=ledger_rows(ledger)
        if before['pending_requests'] or before['unresolved_requests']!=16:raise ValueError('New unresolved/pending usage blocks batch')
        with ledger._db() as db:
            if db.execute('SELECT 1 FROM requests WHERE scope=?',(a.scope,)).fetchone():raise ValueError('This scope was already attempted; no automatic retry')
        record=load_snapshot(a.accepted_history,sha(a.accepted_history))
        if len(record['accepted_unknown_requests'])!=16:raise ValueError('Prior explicit authorization must bind exactly 16 existing rows')
        record.update(user_authorization='2026-11-26 user accepted the existing 16 unknowns and requested automatic next steps',
                      parent_authorization_sha256=sha(a.accepted_history),allowed_scope=a.scope,
                      maximum_new_requests=3,estimated_stop_threshold_rmb=.25,registry_sha256=sha(a.cases/'registry.json'))
        profile=a.private_output/'accepted_history.json';save(profile,record)
        account=AcceptedHistoryAccount(ledger,a.scope,str(run),profile,sha(profile));account.preflight()
        env=os.environ.copy()
        for key in list(env):
            if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_') or key.startswith('SUPERAGENT_ACCEPTED_'):
                env.pop(key)
        env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
                   SUPERAGENT_TASK_BUDGET_ID=a.scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(profile.resolve()),
                   SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(profile),SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',
                   SUPERAGENT_OFFLINE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        destination=run/'delivery/predictions.csv'
        command=[sys.executable,'-m','unified_agent','solve',str((a.cases/row['package']).resolve()),
                 '--prediction-input',str((a.cases/row['query']).resolve()),'--task-interpretation','proactive',
                 '--task-analysis','local','--protected-self-review','--fit-limit','2','--time-limit','240',
                 '--run-dir',str(run.resolve()),'--output',str(destination.resolve())]
        save(a.output/'invocation.json',dict(command=command,configuration={k:v for k,v in env.items() if k in (
            'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_ACCEPTED_HISTORY_SHA256','SUPERAGENT_CAMPUS_DIRECT_IP','SUPERAGENT_OFFLINE')},
            key_file_configured=True,ledger=a.ledger.as_posix(),accepted_unknown_requests=16,profile_sha256=sha(profile),
            registry_sha256=sha(a.cases/'registry.json'),package_sha256=row['package_sha256'],query_sha256=row['query_sha256']))
        status='validated_no_requests'
        if not a.validate_only:
            # One subprocess invocation only. Even an ambiguous failure is retained, never retried.
            status='failed'
            logpath=a.private_output/'solve.log'
            with logpath.open('w',encoding='utf-8') as stream:
                proc=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                    timeout=310)
            returncode=proc.returncode
            if returncode:raise RuntimeError('Production solve exited '+str(returncode)+'; inspect private solve.log')
            selection=read(run/'agent/final_selection.json')
            if not selection.get('refit'):raise ValueError('Expected a complete bounded refit')
            semantic=read(run/'agent/semantic_contract.json')
            if semantic.get('probe')!='passed' or not (run/'agent/semantic_artifact.json').is_file():raise ValueError('No fresh accepted semantic interpretation')
            # Only inspect output formatting after execution. Development labels are never opened.
            oracle=read(a.cases/'V1.oracle.json');recipe=oracle['expected']['output']
            with destination.open(encoding=recipe['encoding'],newline='') as stream:
                records=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
            if len(records)!=row['prediction_rows'] or list(records[0])!=[c['name'] for c in recipe['columns']]:
                raise ValueError('Export header/row mismatch')
            if [r[recipe['columns'][0]['name']] for r in records]!=[f'{i:06d}' for i in range(len(records))]:
                raise ValueError('Original prediction IDs/order changed')
            scores=[float(r['PositiveScore']) for r in records]
            if not all(0<=x<=1 for x in scores):raise ValueError('Invalid probabilities')
            delivery=dict(rows=len(records),model_sha256=selection['model_sha256'],prediction_sha256=sha(destination),
                          family=selection['family'],refit=True,fresh_semantic_artifact=True)
            save(a.output/'final_selection.json',selection)
            save(a.output/'semantic_contract.json',semantic)
            status='passed'
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc)[:900])
    fitfile=run/'agent/classifier_fits.json'
    if fitfile.is_file():
        accounting=read(fitfile);used=accounting['attempted_classifier_fits'];save(a.output/'classifier_fits.json',accounting)
    after=ledger.project_summary() if ledger else None;new=[];unchanged_old=None
    if ledger:
        current=ledger_rows(ledger);unchanged_old=all(current.get(k)==v for k,v in old_rows.items())
        with ledger._db() as db:
            new=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(a.scope,))]
        save(a.output/'ledger_delta.json',new)
        if not unchanged_old or used>2 or len(new)>3:status='failed'
        if after['unresolved_requests']!=16 or after['pending_requests']:status='stopped_new_unresolved'
    unchanged_source=all(sha(root/name)==digest for name,digest in identity.items())
    if not unchanged_source:status='failed'
    result=dict(status=status,error=error,subprocess_returncode=returncode,delivery=delivery,
                actual_classifier_fits=used,real_API_requests=len(new),known_course_cost_rmb=sum(r['course_units'] or 0 for r in new)/1e9,
                new_unresolved_requests=sum(r['status']!='known' for r in new),ledger_before=before,ledger_after=after,
                historical_rows_unchanged=unchanged_old,source_unchanged=unchanged_source,elapsed_seconds=time.monotonic()-clock,
                course_budget_compliance='unverified_history',quality_evaluation=False,
                evidence_boundary='production fresh-task deployment on a previously used development input; not arbitrary unknown-task success')
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return status not in ('passed','validated_no_requests')


if __name__=='__main__':raise SystemExit(main())
