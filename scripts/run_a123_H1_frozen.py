"""Prepare/seal H1 using a source-bound registered snapshot; no extra teammate consent form."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import ROOT,BASE,environment,child,physical

POLICY=dict(Fstar='L0',arms=['L0','R0'],outer_seeds=['61411','61421'],catalog='standard',
            parameter_semantics='off',seed=707,fit_limit=11,trial_timeout=300,time_limit=1200)
PAID='''
import json,sys
from pathlib import Path
from scripts.campus_direct import campus_resolution,CAMPUS_IP
from tools.ab_registered_run import run
info=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
with campus_resolution(CAMPUS_IP):
    raise SystemExit(run(Path(info['manifest']),info['manifest_sha256'],info['command']))
'''


def check_freeze(receipt,digest,registration):
    if not receipt or not digest:return 'Common freeze receipt has not been supplied'
    if sha(receipt)!=digest:raise ValueError('Freeze receipt bytes changed')
    record=read(receipt)
    status=record.get('status')
    confirmation=(status=='registered_experiment_snapshot' or
                  status=='jointly_frozen' and record.get('A_confirmation') and record.get('B_confirmation'))
    if (not confirmation or record.get('policy')!=POLICY or record.get('source_files')!=registration['source_files'] or
        record.get('input_registration_sha256')!=registration['input_registration_sha256']):
        raise ValueError('Shared confirmations/source/policy/input scope do not match')
    return None


def prepare(output):
    source=hashes(ROOT);original=ROOT/'research/a114_H1_inputs/results.json';registered=read(original)['training_inputs']
    entries={}
    for seed,row in registered.items():
        folder=BASE/'runs/a114_H1_inputs'/seed
        if sha(folder/'train.zip')!=row['train_archive_sha256'] or sha(folder/'inner_splits.json')!=row['inner_manifest_sha256']:
            raise ValueError('Previously sealed H1 input changed')
        entries[seed]=dict(training=str(folder/'train.zip'),manifest=str(folder/'inner_splits.json'),
            training_sha256=row['train_archive_sha256'],manifest_sha256=row['inner_manifest_sha256'],
            capabilities=row['capabilities'],registered_splits=row['registered_splits'],rows=row['train_rows'])
    registration=dict(source_files=source,input_registration_sha256=sha(original),entries=entries,policy=POLICY,
                      outer_labels_opened=False,performance_seen=False)
    snapshot=output/'experiment_snapshot.json'
    save(snapshot,dict(status='registered_experiment_snapshot',
        authority='User chose latest B delivery and authorized A integration; this is a run snapshot, not final submission freeze',
        policy=POLICY,source_files=source,input_registration_sha256=sha(original)))
    registration['snapshot_sha256']=sha(snapshot)
    save(output/'registration.json',registration)
    save(output/'freeze_receipt_TEMPLATE.json',dict(status='proposal_only_not_frozen',A_confirmation=None,B_confirmation=None,
        policy=POLICY,source_files=source,input_registration_sha256=sha(original)))
    return dict(status='prepared_registered_snapshot',actual_classifier_fits=0,real_API_requests=0,
                outer_scores_computed=0,inputs_previously_validated=True)


def receipt_for(a,registration):
    if a.freeze_receipt:return a.freeze_receipt,a.freeze_sha256
    return a.prepared/'experiment_snapshot.json',registration.get('snapshot_sha256')


def train(a):
    registration=read(a.prepared/'registration.json')
    if hashes(ROOT)!=registration['source_files']:raise ValueError('Source changed since registration')
    reason=check_freeze(*receipt_for(a,registration),registration)
    if reason:return dict(status='blocked_before_training',reason=reason,actual_classifier_fits=0,real_API_requests=0)
    entry=registration['entries'][a.outer]
    if sha(Path(entry['training']))!=entry['training_sha256'] or sha(Path(entry['manifest']))!=entry['manifest_sha256']:
        raise ValueError('Registered H input changed')
    paid=a.arm=='R0';planner='raw-history' if paid else 'local';env=environment(a.output)
    env['AB_A29_RUN_NAME']=a.outer+'_'+a.arm
    command=['run',entry['training'],'--run-dir',str(a.output/'run'),'--planner',planner,'--scheduler','ab-fixed',
        '--portfolio','ab-v1','--preference-catalog','standard','--parameter-semantics','off','--catalog-scope','full',
        '--fit-limit','11','--seed','707','--trial-timeout','300','--time-limit','1200','--partial-delivery','checkpoint']
    ledger=None;before=None
    if paid:
        if not a.ledger or not a.api_key_file or not a.ledger.exists():raise ValueError('Existing shared ledger/key paths required')
        # Never create or replace a ledger, nor import requests from a receipt.
        from superagent.task_budget import TaskBudgetLedger
        from superagent.research_budget import research_preflight
        ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary()
        if before['requests']<357 or before['unresolved_requests']>11 or before['pending_requests']:
            raise ValueError('Returned real history/new unknown still unresolved; zero new API requests')
        if not a.writer_receipt or not a.writer_sha256 or sha(a.writer_receipt)!=a.writer_sha256:
            raise ValueError('Exclusive shared-ledger writer receipt required')
        writer=read(a.writer_receipt)
        if (writer.get('owner')!='A' or writer.get('ledger_sha256')!=sha(a.ledger) or
            writer.get('project_summary')!=before or writer.get('window_status')!='exclusive_active'):
            raise ValueError('Writer handoff identity differs')
        research_preflight(ledger,'H1_transfer',.30)
        command+=['--api-key-file',str(a.api_key_file),'--budget-ledger',str(a.ledger),'--task-budget-id','H1_transfer',
                  '--llm-budget-rmb','.30','--history-policy','incremental-research','--proposal-contract','calibrated-v2']
        env['SUPERAGENT_OFFLINE']='0'
    info=dict(manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],command=command,
              project_usage_before=before)
    save(a.output/'command.json',info)
    if a.validate_only:return dict(status='validated_before_training',actual_classifier_fits=0,real_API_requests=0)
    launch=([sys.executable,'-c',PAID,str(a.output/'command.json')] if paid else
            [sys.executable,'-m','tools.ab_registered_run','--split-manifest',entry['manifest'],
             '--split-sha256',entry['manifest_sha256'],'--',*command])
    process=child(launch,a.output/'train.log',env,1260,allow_failure=True)
    outcome=read(a.output/'run/run_outcome.json');manifest=read(a.output/'run/run_manifest.json')
    count=len(physical(a.output))
    if count!=outcome['classifier_fit_accounting']['attempted_classifier_fits'] or count>11:
        raise ValueError('Physical classifier fit count mismatch')
    if manifest['capabilities']!=entry['capabilities']:raise ValueError('Actual full H menu changed')
    if hashes(ROOT)!=registration['source_files']:raise ValueError('Source changed during H training')
    selection=read(a.output/'run/final_selection.json') if outcome['model_available'] else None
    seal=None
    if selection:
        if sha(a.output/'run'/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model changed')
        seal=dict(outer=a.outer,arm=a.arm,selection=selection,source_files=registration['source_files'],
                  training_sha256=entry['training_sha256'],complete=outcome['requested_planner_completed'])
        save(a.output/'model_seal.json',seal)
    save(a.output/'results.json',dict(process=process,outcome=outcome,model_seal=seal,
        project_usage_after=ledger.project_summary() if ledger else None))
    return dict(status='sealed' if outcome['requested_planner_completed'] and not process['returncode'] else 'stopped',
        actual_classifier_fits=count,real_API_requests=outcome['new_api_requests'],automatic_retry=False,
        outer_scores_computed=0,reason='All four H1 paths must be sealed before evaluation')


def score(a):
    registration=read(a.prepared/'registration.json')
    reason=check_freeze(*receipt_for(a,registration),registration)
    if reason:return dict(status='blocked_before_evaluation',reason=reason,actual_classifier_fits=0,real_API_requests=0,outer_labels_accessed=False)
    if hashes(ROOT)!=registration['source_files']:raise ValueError('Scoring source differs from frozen models')
    if not a.matrix or not a.matrix_sha256 or sha(a.matrix)!=a.matrix_sha256:
        raise ValueError('Signed four-path matrix receipt required before any outer-label access')
    matrix=read(a.matrix);required={s+'_'+arm for s in POLICY['outer_seeds'] for arm in POLICY['arms']}
    if set(matrix)!=required:raise ValueError('All four frozen H1 paths are required')
    sealed={}
    for key,location in matrix.items():
        folder=Path(location);seal=read(folder/'model_seal.json');completion=read(folder/'completion.json')
        seed,arm=key.split('_')
        if (completion['status']!='sealed' or not seal['complete'] or seal['outer']!=seed or seal['arm']!=arm or
            seal['source_files']!=registration['source_files'] or
            seal['training_sha256']!=registration['entries'][seed]['training_sha256']):
            raise ValueError('Incomplete/mismatched matrix path: '+key)
        selection=seal['selection']
        if sha(folder/'run'/selection['model_file'])!=selection['model_sha256']:
            raise ValueError('Frozen matrix model changed')
        sealed[key]=folder
    # All planned paths passed before decoding outer text or reading outer labels.
    registry_file=ROOT/'research/a12_a16_day/a13_H1_publisher/registration.json'
    if sha(registry_file)!=read(ROOT/'research/a114_H1_inputs/protocol.json')['original_registration_sha256']:
        raise ValueError('Original outer registration changed')
    registry=read(registry_file)['H1'];publisher=BASE/'runs/a13_H1_publisher/train.parquet'
    if sha(publisher)!=registry['source']['sha256']:raise ValueError('Publisher source changed')
    from scripts import a38_predict_guard
    import pyarrow.parquet as pq
    from scripts.run_b1_05_pair_pilot import pair_package
    results=[];env=environment(a.output)
    for seed in POLICY['outer_seeds']:
        ids=[registry['selected_original_ids'][i] for i in registry['partitions'][seed]['held']]
        selected=pq.read_table(publisher,columns=['id','sentence1','sentence2','label'],filters=[('id','in',ids)]).to_pylist()
        lookup={r['id']:r for r in selected}
        if set(lookup)!=set(ids) or len(selected)!=len(ids):raise ValueError('Original outer row identity differs')
        target=a.output/seed;target.mkdir();features=target/'features.zip'
        pair_package(features,[(lookup[i]['sentence1'],lookup[i]['sentence2']) for i in ids],None,'symmetric','test')
        labels=[str(int(lookup[i]['label'])) for i in ids]
        for arm in POLICY['arms']:
            key=seed+'_'+arm;run=sealed[key]/'run';destination=target/(arm+'.csv')
            child([sys.executable,'-m','scripts.a38_predict_guard',str(features),'--run-dir',str(run),
                '--output',str(destination)],target/(arm+'.log'),env,180)
            with destination.open(encoding='utf-8',newline='') as stream:predictions=list(csv.DictReader(stream))
            if [int(r['row_id']) for r in predictions]!=list(range(len(ids))):raise ValueError('Outer prediction order changed')
            correct=sum(r['prediction']==label for r,label in zip(predictions,labels,strict=True))
            results.append(dict(outer=seed,arm=arm,rows=len(ids),correct=correct,accuracy=correct/len(ids),
                                predictions_sha256=sha(destination),all_models_frozen_before_scoring=True))
    save(a.output/'results.json',results)
    return dict(status='scored_frozen_H1',paths=4,actual_classifier_fits=0,real_API_requests=0,
                outer_labels_accessed=True,H_not_used_for_selection=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','train','score'])
    for name in ('output','prepared','freeze-receipt','ledger','api-key-file','writer-receipt','matrix'):p.add_argument('--'+name,type=Path,required=name=='output')
    for name in ('freeze-sha256','writer-sha256','matrix-sha256'):p.add_argument('--'+name)
    p.add_argument('--outer',choices=POLICY['outer_seeds']);p.add_argument('--arm',choices=POLICY['arms'])
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A123-'+a.phase,runner_sha256=sha(Path(__file__)),source_files=hashes(ROOT),
        plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),
        freeze_receipt_sha256=sha(a.freeze_receipt) if a.freeze_receipt else None,
        maximum_classifier_fits=0 if a.phase!='train' or a.validate_only else 11,maximum_API_requests=3 if a.arm=='R0' else 0,
        automatic_retry=False,outer_labels_accessed=False))
    try:
        result=prepare(a.output) if a.phase=='prepare' else score(a) if a.phase=='score' else train(a)
        save(a.output/'completion.json',dict(**result,elapsed_seconds=time.monotonic()-start));print(json.dumps(result),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,
            observed_classifier_fit_entries=len(physical(a.output)),real_API_requests=None,
            note='Read run outcome/true ledger if a paid child was started; null is not zero'))
        raise


if __name__=='__main__':main()
