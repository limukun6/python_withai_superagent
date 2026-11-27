"""SA23: real JSON-tool delivery recovery with an explicitly accepted 17-row history."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa21_context_live import clean_env,execute

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/SA23_PAID_DELIVERY_VALIDATION_PLAN_20261127_zh.md'


def identity():
    paths=list((ROOT/'superagent').glob('*.py'))+[ROOT/'unified_agent.py',Path(__file__),ROOT/'tools/import_semantic_artifact.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def prepare(a):
    from tools.import_semantic_artifact import migrate
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Required checks must pass')
    if any(sha(ROOT/n)!=v for n,v in read(a.checks/'protocol.json')['source_sha256'].items()):raise ValueError('Checked source changed')
    cases=a.output/'cases';cases.mkdir();registry=[]
    for source in read(a.source/'cases/registry.json'):
        old=source['id'];name={'C1':'V1','C2':'V2'}[old]
        package=cases/(name+'.zip');reference=cases/(name+'.reference.zip');query=cases/(name+'.late.zip')
        for key,path in [('package',package),('query',reference)]:
            if sha(a.source/'cases'/source[key])!=source[key+'_sha256']:raise ValueError('Original input changed')
            shutil.copyfile(a.source/'cases'/source[key],path)
        artifact=a.private_output/(name+'.semantic_artifact.json')
        metadata=migrate(package,reference,a.source_runs/old/'run/agent/semantic_contract.json',a.ledger,artifact)
        save(a.output/(name+'.migration.json'),metadata)
        with zipfile.ZipFile(reference) as z:
            if name=='V1':
                rows=[json.loads(s) for s in z.read('arrivals.jsonl').decode().splitlines()]
                stream=io.StringIO(newline='');writer=csv.writer(stream)
                writer.writerow(['delivery_token']+[f'measurement_{i:02d}' for i in range(30)])
                writer.writerows([r['specimen_key']]+[r[f'read_{i:02d}'] for i in range(30)] for r in rows)
                members={'arrivals_23.csv':stream.getvalue(),'README.md':(
                    'This is a later prediction batch for the same frozen classifier. arrivals_23.csv replaces arrivals.jsonl. '
                    'Each numeric measurement_NN field corresponds to canonical training measure_NN by the same two-digit suffix '
                    '00 through 29. The only remaining field is the original record identifier. Its exact header is not supplied '
                    'here; inspect the table header. Case must preserve that field including leading zeros. '
                    'Retain the frozen tab/BOM output, Decision labels and BenignScore probability of B. Do not train or change task meanings.')}
            else:
                features=np.load(io.BytesIO(z.read('features.npy')),allow_pickle=False)
                stream=io.BytesIO();np.savez_compressed(stream,reference_bundle_9c=np.zeros_like(features),observation_bundle_7f=features)
                members={'dual_bundle.npz':stream.getvalue(),'README.md':(
                    'This is a later prediction batch for the same frozen classifier. dual_bundle.npz replaces features.npy. '
                    'There are two same-shape numeric matrices. The array whose actual key starts with observation_bundle_ '
                    'contains the new observations in unchanged row/column order; the key starting reference_bundle_ '
                    'is an instrument reference and must not be predicted. Inspect the container for the full key names. '
                    'There are no labels or original IDs. Preserve Index;Class, zero-based rows and frozen class codes; do not retrain.')}
        with zipfile.ZipFile(query,'w') as z:
            for member,body in members.items():z.writestr(member,body)
        oracle=read(a.source/'cases'/(old+'.oracle.json'));save(cases/(name+'.oracle.json'),oracle)
        registry.append(dict(id=name,source_case=old,package=package.name,reference=reference.name,query=query.name,
            package_sha256=sha(package),reference_sha256=sha(reference),query_sha256=sha(query),
            oracle_sha256=sha(cases/(name+'.oracle.json')),artifact=str(artifact.resolve()),artifact_sha256=sha(artifact)))
    save(cases/'registry.json',registry)
    save(a.output/'protocol.json',dict(batch='SA23-paid-JSON-delivery-validation',base='15a79b4',source_sha256=identity(),plan_sha256=sha(PLAN),
        registry_sha256=sha(cases/'registry.json'),maximum_actual_fits=4,maximum_real_API=6,per_path_fit_limit=2,
        per_path_scope_requests=3,estimated_stop_threshold_rmb_per_path=.30,
        scope='controlled later-format conditions, explicit validated interpretation reuse; not raw unseen solve or quality comparison'))
    save(a.output/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_API_requests=0))
    print(dict(status='prepared',cases=[r['id'] for r in registry]))


def run(a):
    from superagent.task_budget import TaskBudgetLedger
    from superagent.accepted_history import AcceptedHistoryAccount,load_snapshot
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    protocol=read(a.prepared/'protocol.json');cases=a.prepared/'cases';start=time.monotonic()
    if protocol['source_sha256']!=identity() or protocol['plan_sha256']!=sha(PLAN):raise ValueError('Frozen source/plan changed')
    if sha(cases/'registry.json')!=protocol['registry_sha256']:raise ValueError('Registry changed')
    registry=read(cases/'registry.json')
    for row in registry:
        for key in ('package','reference','query'):
            if sha(cases/row[key])!=row[key+'_sha256']:raise ValueError('Input changed')
        if sha(row['artifact'])!=row['artifact_sha256']:raise ValueError('Migrated artifact changed')
    save(a.output/'protocol.json',dict(protocol,prepared_protocol_sha256=sha(a.prepared/'protocol.json'),scope_prefix=a.scope))
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger)
    if before['unresolved_requests']!=17 or before['pending_requests']:raise ValueError('New unresolved usage blocks batch')
    prior=load_snapshot(a.accepted_history,sha(a.accepted_history));results=[];fatal=None
    for row in registry:
        name=row['id'];scope=a.scope+'_'+name.lower();out=a.output/name;private=a.private_output/name
        out.mkdir();private.mkdir();run_dir=private/'run';clock=time.monotonic();result=dict(id=name,status='failed',false_completion=False)
        try:
            with ledger._db() as db:
                if db.execute('SELECT 1 FROM requests WHERE scope=?',(scope,)).fetchone():raise ValueError('Attempted scope cannot be retried')
            profile=dict(prior,user_authorization='2026-11-27 user authorized ignoring unknown usage and continuing paid research; frozen SA23 two-path batch, current 17 unknown rows accepted',
                parent_authorization_sha256=sha(a.accepted_history),allowed_scope=scope,maximum_new_requests=3,
                estimated_stop_threshold_rmb=.30,registry_sha256=protocol['registry_sha256'])
            path=private/'accepted_history.json';save(path,profile)
            AcceptedHistoryAccount(ledger,scope,str(run_dir),path,sha(path)).preflight()
            env=clean_env();env.update(SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file.resolve()),SUPERAGENT_PLANNER_LEDGER=str(a.ledger.resolve()),
                SUPERAGENT_TASK_BUDGET_ID=scope,SUPERAGENT_ACCEPTED_HISTORY_FILE=str(path.resolve()),
                SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(path),SUPERAGENT_CAMPUS_DIRECT_IP='10.121.10.250',SUPERAGENT_OFFLINE='0')
            package=str((cases/row['package']).resolve());reference=str((cases/row['reference']).resolve());query=str((cases/row['query']).resolve())
            train=[sys.executable,'-m','unified_agent','train',package,'--prediction-input',reference,'--semantic-artifact-from',row['artifact'],
                '--run-dir',str(run_dir.resolve()),'--fit-limit','2','--time-limit','360']
            def predict(source,destination):return [sys.executable,'-m','unified_agent','predict',source,'--run-dir',str(run_dir.resolve()),'--output',str(destination.resolve())]
            original=private/'reference/predictions.csv';delivery=private/'delivery/predictions.csv';replay=private/'replay/predictions.csv'
            save(out/'invocation.json',dict(train=train,reference=predict(reference,original),late=predict(query,delivery),replay=predict(query,replay),scope=scope,
                training_interpretation='explicit validated historical artifact; zero new interpretation requests',injected_fault=False))
            if execute(train,env,private/'train.log'):raise RuntimeError('Formal cached-interpretation training failed')
            model=run_dir/'agent';selection=read(model/'final_selection.json')
            if not selection.get('refit'):raise ValueError('No final trained model')
            frozen={str(p):sha(p) for p in [model/selection['model_file'],model/'task_spec.json',model/'final_selection.json',model/'classifier_fits.json']}
            if execute(predict(reference,original),env,private/'reference.log',60):raise RuntimeError('Reference prediction failed')
            with ledger._db() as db:before_predict=db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(scope,)).fetchone()[0]
            if before_predict:raise ValueError('Explicit cached training/reference unexpectedly called API')
            if execute(predict(query,delivery),env,private/'late.log',180):raise RuntimeError('Autonomous late prediction recovery failed')
            with ledger._db() as db:after_predict=db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(scope,)).fetchone()[0]
            if execute(predict(query,replay),env,private/'replay.log',60):raise RuntimeError('Independent cached prediction failed')
            with ledger._db() as db:after_replay=db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(scope,)).fetchone()[0]
            unchanged=frozen=={name:sha(name) for name in frozen}
            if not unchanged or sha(original)!=sha(delivery) or sha(original)!=sha(replay):
                result['false_completion']=True
                raise ValueError('Frozen files or predictions differ')
            if after_replay!=after_predict:raise ValueError('Cached replay spent extra API')
            recovery=list((run_dir/'delivery_recovery').glob('*/completion.json'))
            if len(recovery)!=1 or read(recovery[0])['status']!='adapter_verified':raise ValueError('Expected one actual failed-input recovery')
            manifest=read(delivery.parent/'prediction_manifest.json')
            if not manifest.get('prediction_adapter_sha256'):raise ValueError('No verified adapter in delivery receipt')
            from superagent.prediction_adapter import load_adapter
            from superagent.data import load_prediction_features
            from scripts.sa7_multiturn_cases import array_fingerprint
            original_spec=read(model/'task_spec.json')
            adapted=load_adapter(recovery[0].parent/'prediction_adapter.json',query,model,original_spec,selection)
            features=load_prediction_features(query,original_spec['model_kind'],adapted)
            oracle=read(cases/(name+'.oracle.json'))
            if array_fingerprint(features)!=oracle['expected']['prediction_features_sha256']:
                result['false_completion']=True
                raise ValueError('Adapted features differ from frozen oracle')
            save(out/'recovery_completion.json',read(recovery[0]));save(out/'recovery_state.json',read(recovery[0].parent/'recovery_state.json'))
            save(out/'prediction_adapter.json',read(recovery[0].parent/'prediction_adapter.json'))
            save(out/'prediction_manifest.json',manifest);save(out/'frozen_files.json',dict(before=frozen,after={name:sha(name) for name in frozen}))
            result.update(status='passed',false_completion=False,prediction_rows=manifest['prediction_count'],model_sha256=selection['model_sha256'],
                frozen_files_unchanged=True,predictions_equal_in_three_processes=True,training_and_reference_API=before_predict,
                real_tool_observations=len(read(recovery[0].parent/'recovery_state.json')['observed_evidence']),
                decision_source=read(recovery[0]).get('decision_source','real_LLM'),feature_fingerprint_verified=True,recovery_new_fits=0,cached_replay_new_API=after_replay-after_predict,recovery_API=after_predict)
        except Exception as issue:result.update(error=dict(type=type(issue).__name__,message=str(issue)[:700]))
        fit=run_dir/'agent/classifier_fits.json';account=read(fit) if fit.is_file() else {'attempted_classifier_fits':0}
        save(out/'classifier_fits.json',account);result['actual_classifier_fits']=account['attempted_classifier_fits']
        calls=[]
        for log in (run_dir/'delivery_recovery').glob('*/llm_usage.jsonl'):
            calls += [json.loads(s) for s in log.read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')=='llm_call']
        save(out/'metered_usage.json',calls)
        with ledger._db() as db:delta=[dict(r) for r in db.execute('SELECT * FROM requests WHERE scope=?',(scope,))]
        save(out/'ledger_delta.json',delta)
        result.update(real_API_requests=len(delta),known_course_cost_rmb=sum(r['course_units'] or 0 for r in delta)/1e9,
                      elapsed_seconds=time.monotonic()-clock,human_interventions=0)
        after=ledger.project_summary()
        if after['unresolved_requests']!=17 or after['pending_requests']:fatal='new_unresolved_usage'
        if result['actual_classifier_fits']>2 or len(delta)>3:fatal='resource_limit_exceeded'
        save(out/'completion.json',result);results.append(result);print(json.dumps(result),flush=True)
        if fatal:break
    current=ledger_rows(ledger);unchanged=all(current.get(k)==v for k,v in old.items())
    result=dict(status='stopped' if fatal else 'passed' if len(results)==2 and all(r['status']=='passed' for r in results) else 'failed',cases=results,
        actual_classifier_fits=sum(r['actual_classifier_fits'] for r in results),real_API_requests=sum(r['real_API_requests'] for r in results),
        known_course_cost_rmb=sum(r['known_course_cost_rmb'] for r in results),elapsed_seconds=time.monotonic()-start,
        ledger_before=before,ledger_after=ledger.project_summary(),historical_rows_unchanged=unchanged,source_unchanged=protocol['source_sha256']==identity(),fatal=fatal)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare')
    for key in ('output','private-output','checks','source','source-runs','ledger'):p.add_argument('--'+key,type=Path,required=True)
    p=sub.add_parser('run')
    for key in ('prepared','output','private-output','ledger','key-file','accepted-history'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scope',default='sa23_delivery_20261127_01');args=parser.parse_args()
    return prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':main()
