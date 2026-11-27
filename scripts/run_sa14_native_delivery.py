"""Prepare then run two registered production native-input tasks; no interactive intervention."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa12_live_entry import ledger_rows
from scripts.sa7_multiturn_cases import array_fingerprint


def identity(root):
    paths=list((root/'superagent').glob('*.py'))+[root/'unified_agent.py',Path(__file__),root/'scripts/sa14_native_cases.py']
    return {p.relative_to(root).as_posix():sha(p) for p in paths}


def grade(task,query,expected):
    from superagent.data import load_prediction_features
    from superagent.metrics import MetricSpec
    from tools.submission_equivalence import equivalent_csv_recipe
    metric=MetricSpec.for_task(task);x=load_prediction_features(query,task.kind,task.spec)
    checks=dict(kind=task.kind==expected['kind'],metric=metric.name==expected['metric'],classes=list(metric.labels)==expected['classes'],
        training_rows=len(task.y_train)==expected['training_rows'],training_features=array_fingerprint(task.X_train)==expected['training_features_sha256'],
        training_labels=array_fingerprint(task.y_train)==expected['training_labels_sha256'],
        prediction_shape=list(x.shape)==expected['prediction_shape'],prediction_features=array_fingerprint(x)==expected['prediction_features_sha256'],
        output=equivalent_csv_recipe(task.spec.get('output_recipe'),expected['output'],list(metric.labels)))
    return dict(complete=all(checks.values()),checks=checks)


def validate_registry(cases):
    registry=read(cases/'registry.json')
    if [row['id'] for row in registry]!=['N1','N2']:raise ValueError('Expected the two registered native cases')
    for row in registry:
        for field,hashfield in [('package','package_sha256'),('query','query_sha256')]:
            if sha(cases/row[field])!=row[hashfield]:raise ValueError('Registered native package changed')
        if sha(cases/(row['id']+'.oracle.json'))!=row['oracle_sha256']:raise ValueError('Registered oracle changed')
    return registry


def prepare(a):
    from scripts.sa14_native_cases import build
    from superagent.ab_run import Executor
    from superagent.format_sources import package_files
    from superagent.native_semantics import compile_native
    from superagent.semantic_contract import evidence_workspace
    registry=build(a.cases);rows=[]
    with patch.object(Executor,'worker',side_effect=AssertionError('Preparation forbids fitting')),patch.object(socket,'create_connection',side_effect=AssertionError('Preparation forbids network')):
        for row in registry:
            oracle=read(a.cases/(row['id']+'.oracle.json'))
            with package_files(a.cases/row['package']) as files,package_files(a.cases/row['query']) as query:
                task=compile_native(files,evidence_workspace(files),oracle['answer'],prediction_files=query)
            result=grade(task,a.cases/row['query'],oracle['expected'])
            rows.append(dict(case=row['id'],**result,scope='developer-scripted proposal preparation; not a real model response'))
    save(a.output/'registry.json',registry)
    return dict(status='prepared' if all(r['complete'] for r in rows) else 'failed',cases=rows,
                actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0)


def live(a,root):
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    from superagent.semantic_artifact import load_artifact
    from superagent.task_budget import TaskBudgetLedger
    registry=validate_registry(a.cases)
    checked=read(a.checks/'completion.json')
    if checked['status']!='passed' or not checked.get('source_unchanged'):raise ValueError('Offline native checks not passed')
    for name,digest in read(a.checks/'protocol.json')['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Checked implementation changed')
    if read(a.preparation/'completion.json')['status']!='prepared' or read(a.preparation/'registry.json')!=registry:
        raise ValueError('Case preparation not complete or registry changed')
    if not all([a.ledger,a.key_file,a.accepted_history,a.private_output]):raise ValueError('Live mode requires real ledger/key/accepted history/private output')
    a.private_output.mkdir(parents=True,exist_ok=False)
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger)
    if before['unresolved_requests']!=16 or before['pending_requests']:raise ValueError('New unknown/pending blocks live batch')
    original=load_snapshot(a.accepted_history,sha(a.accepted_history));results=[];start=time.monotonic()
    for row in registry:
        name=row['id'];scope='sa14_native_20261127_'+name.casefold();run=a.private_output/name
        with ledger._db() as db:
            if db.execute('SELECT 1 FROM requests WHERE scope=?',(scope,)).fetchone():raise ValueError('Registered scope already attempted; no automatic retry')
        record=dict(original,allowed_scope=scope,maximum_new_requests=3,estimated_stop_threshold_rmb=.25,
            parent_authorization_sha256=sha(a.accepted_history),registry_sha256=sha(a.cases/'registry.json'),
            user_authorization='User accepted existing 16 unknown rows and requested automatic next steps; bounded SA14 registered scope')
        profile=a.private_output/(name+'_accepted_history.json');save(profile,record)
        account=AcceptedHistoryAccount(ledger,scope,str(run),profile,sha(profile));account.preflight()
        result=dict(case=name,status='failed',actual_classifier_fits=0,real_API_requests=0)
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
        env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
            SUPERAGENT_TASK_BUDGET_ID=scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(profile.resolve()),
            SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(profile),SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',
            SUPERAGENT_OFFLINE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        destination=run/'delivery/predictions.csv'
        command=[sys.executable,'-m','unified_agent','solve',str((a.cases/row['package']).resolve()),
            '--prediction-input',str((a.cases/row['query']).resolve()),'--task-interpretation','proactive',
            '--task-analysis','local','--protected-self-review','--fit-limit','2','--time-limit','240',
            '--run-dir',str(run.resolve()),'--output',str(destination.resolve())]
        evidence=a.output/name;evidence.mkdir();save(evidence/'invocation.json',dict(command=command,scope=scope,
            profile_sha256=sha(profile),registry_sha256=sha(a.cases/'registry.json'),stdin='closed',intervention='none'))
        clock=time.monotonic()
        try:
            with (a.private_output/(name+'_solve.log')).open('w',encoding='utf-8') as stream:
                proc=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=310)
            if proc.returncode:raise RuntimeError('Official solve returned '+str(proc.returncode)+'; see private log')
            selection=read(run/'agent/final_selection.json')
            if not selection.get('refit'):raise ValueError('Only partial delivery, not a complete native-task refit')
            # External evaluator runs only after execution. It cannot repair a plan or select a model.
            task=load_artifact(run/'agent/semantic_artifact.json',a.cases/row['package'],a.cases/row['query'])
            oracle=read(a.cases/(name+'.oracle.json'));verification=grade(task,a.cases/row['query'],oracle['expected'])
            save(evidence/'semantic_grade.json',verification)
            if not verification['complete']:raise ValueError('Completed run differs from registered semantic contract')
            recipe=oracle['expected']['output']
            with destination.open(encoding=recipe['encoding'],newline='') as stream:predictions=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
            if len(predictions)!=row['prediction_rows'] or list(predictions[0])!=['Row','Decision']:
                raise ValueError('Actual serialized export header/row mismatch')
            if [r['Row'] for r in predictions]!=[str(i+1) for i in range(len(predictions))]:raise ValueError('Original declared order/index changed')
            if not {r['Decision'] for r in predictions}<=set(map(str,oracle['expected']['classes'])):raise ValueError('Exported class outside training domain')
            if recipe['encoding']=='utf-8-sig' and not destination.read_bytes().startswith(b'\xef\xbb\xbf'):raise ValueError('Required BOM absent')
            receipt=read(run/'delivery/prediction_manifest.json')
            if receipt['delivery_status']!='completed' or not receipt['refit']:raise ValueError('Prediction receipt not complete')
            result.update(status='passed',family=selection['family'],rows=len(predictions),semantic_grade=verification,
                model_sha256=selection['model_sha256'],prediction_sha256=sha(destination))
            save(evidence/'final_selection.json',selection);save(evidence/'prediction_manifest.json',receipt)
            save(evidence/'semantic_contract.json',read(run/'agent/semantic_contract.json'))
        except Exception as exc:
            result['error']=dict(type=type(exc).__name__,message=str(exc)[:900])
        fitfile=run/'agent/classifier_fits.json'
        if fitfile.is_file():
            fits=read(fitfile);result['actual_classifier_fits']=fits['attempted_classifier_fits'];save(evidence/'classifier_fits.json',fits)
        for filename,kindfield,events in [('events.jsonl','kind',{'semantic_contract_accepted','semantic_contract_rejected','semantic_structure_observed','partial_prior_frozen'}),
                                         ('llm_usage.jsonl','event',{'llm_call','unresolved_call','missing_usage'})]:
            source=run/'agent'/filename
            if source.is_file():save(evidence/filename.replace('.jsonl','.json'),[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if json.loads(s).get(kindfield) in events])
        with ledger._db() as db:new=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(scope,))]
        save(evidence/'ledger_delta.json',new)
        result.update(real_API_requests=len(new),known_course_cost_rmb=sum(r['course_units'] or 0 for r in new)/1e9,
                      new_unresolved_requests=sum(r['status']!='known' for r in new),elapsed_seconds=time.monotonic()-clock)
        if result['actual_classifier_fits']>2 or len(new)>3:raise RuntimeError('Registered physical fit/request limit exceeded')
        results.append(result);save(a.output/'partial_results.json',results);print(json.dumps(result,ensure_ascii=False),flush=True)
        if ledger.project_summary()['unresolved_requests']!=16 or ledger.project_summary()['pending_requests']:break
    after=ledger.project_summary();unchanged=all(ledger_rows(ledger).get(k)==v for k,v in old.items())
    return dict(status='passed' if len(results)==2 and all(r['status']=='passed' for r in results) and unchanged else 'completed_with_failures',
        cases=results,actual_classifier_fits=sum(r['actual_classifier_fits'] for r in results),real_API_requests=sum(r['real_API_requests'] for r in results),
        known_course_cost_rmb=sum(r['known_course_cost_rmb'] for r in results),ledger_before=before,ledger_after=after,
        historical_rows_unchanged=unchanged,course_budget_compliance='unverified_history',quality_evaluation=False,
        elapsed_seconds=time.monotonic()-start,evidence_boundary='two public development repackagings; not independent unknown-task transfer or quality gains')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','live'],required=True)
    for name in ['cases','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['ledger','key-file','accepted-history','private-output']:p.add_argument('--'+name,type=Path)
    p.add_argument('--checks',type=Path,default=Path('research/sa14_native_checks_02'))
    p.add_argument('--preparation',type=Path,default=Path('research/sa14_native_prepare_02'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];source=identity(root)
    save(a.output/'protocol.json',dict(batch='SA14-'+a.mode,base='dee7140',source_sha256=source,
        plan_sha256=sha(root/'docs/batches/SA14_native_semantics_plan_20261127_zh.md'),
        maximum_classifier_fits=0 if a.mode=='prepare' else 4,maximum_real_API_requests=0 if a.mode=='prepare' else 6,
        fit_limit_per_path=2,request_limit_per_path=3,estimated_stop_threshold_rmb=0 if a.mode=='prepare' else .5,
        quality_evaluation=False,prediction_answers_opened=False))
    try:result=prepare(a) if a.mode=='prepare' else live(a,root)
    except Exception as exc:
        result=dict(status='failed',error=dict(type=type(exc).__name__,message=str(exc)[:900]),
                    note='Inspect partial_results and real ledger before any continuation; never assume failed calls are zero or retry')
    result['source_unchanged']=identity(root)==source
    if not result['source_unchanged']:result['status']='failed'
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result['status'] not in ('passed','prepared')


if __name__=='__main__':raise SystemExit(main())
