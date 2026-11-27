"""Recheck SA25 raw models, early/final files, original IDs and immutable real ledger."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa25_joint_delivery import identity,alive
from scripts.run_sa21_context_live import clean_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','live','private','ledger','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--windows-live',type=Path,required=True);p.add_argument('--windows-private',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.data import load_prediction_features
    from superagent.prediction_adapter import load_adapter
    from superagent.semantic_artifact import load_artifact
    ledger=TaskBudgetLedger(a.ledger);protocol=read(a.live/'protocol.json');summary=read(a.live/'completion.json')
    old=read(a.live/'historical_rows_before.json');cases=Path(protocol['cases_root']);rows=[]
    save(a.output/'protocol.json',dict(batch='SA25-independent-raw-audit',script_sha256=sha(Path(__file__)),
        live_completion_sha256=sha(a.live/'completion.json'),maximum_new_classifier_fits=0,maximum_new_campus_API=0))
    for name in ('ordinary','custom'):
        root=a.private/name;model=root/'run/agent';spec=read(model/'task_spec.json');selected=read(model/'final_selection.json')
        early=root/'run/first_delivery';prior=read(early/'agent/final_selection.json')
        first=read(early/'prediction_manifest.json');final=read(root/'delivery/prediction_manifest.json')
        fits=read(model/'classifier_fits.json')
        logs=[json.loads(s) for s in (model/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
        checks=dict(model_hash=sha(model/selected['model_file'])==selected['model_sha256'],
            early_hash=sha(early/'predictions.csv')==first['prediction_sha256'],
            early_model_hash=sha(early/'agent'/prior['model_file'])==prior['model_sha256'],
            early_disclosed=first['delivery_status']=='partial' and prior['refit'] is False,
            final_complete=final['delivery_status']=='completed' and final['delivery_phase']=='selected' and selected['refit'] is True,
            final_replay=sha(root/'delivery/predictions.csv')==sha(root/'replay/predictions.csv')==final['prediction_sha256'],
            fit_records=fits['attempted_classifier_fits']==fits['completed_classifier_fits']==2,
            no_LLM_call=not any(e['event'] in ('llm_call','unresolved_call','missing_usage') for e in logs))
        if name=='ordinary':
            with (root/'delivery/predictions.csv').open(encoding='utf-8',newline='') as stream:records=list(csv.DictReader(stream))
            checks.update(restrictions=set(spec['method_restrictions'])=={'linear_only','no_pretrained_assets','no_external_data'},
                rows=len(records)==24,original_row_order=[r['row_id'] for r in records]==[str(i) for i in range(24)],
                labels=set(r['prediction'] for r in records)<=set(map(str,spec['metric_spec']['labels'])))
        else:
            with (root/'delivery/predictions.csv').open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream,delimiter='\t'))
            checks.update(rows=len(records)==40,header=list(records[0])==['Case','Decision','BenignScore'],
                original_ids=[r['Case'] for r in records]==protocol['custom_expected_ids'],
                classes=set(r['Decision'] for r in records)<={'B','M'},probabilities=all(0<=float(r['BenignScore'])<=1 for r in records))
            task=load_artifact(protocol['artifact'],cases/'custom.zip',cases/'custom.query.zip')
            checks['migrated_interpretation']=task.spec['input_recipe']==spec['input_recipe']
        # Use the shipped exporter, then a relocated -I entry that forbids network and training.
        from tools.export_model_bundle import export
        portable=a.private/'portable'/name;portable.parent.mkdir(exist_ok=True)
        bundle=export(model,portable)
        relocated=a.private/'relocated'/name;relocated.parent.mkdir(exist_ok=True)
        shutil.copytree(bundle,relocated)
        destination=a.private/'relocated_delivery'/name/'predictions.csv';destination.parent.mkdir(parents=True)
        proc=subprocess.run([sys.executable,'-I',str((relocated/'predict.py').resolve()),'--features',str((cases/(name+'.query.zip')).resolve()),
            '--output',str(destination.resolve())],cwd=relocated,env=clean_env(),stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
        (portable/'relocated.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        checks['relocated_isolated_export']=proc.returncode==0 and destination.is_file() and sha(destination)==sha(root/'delivery/predictions.csv')
        save(a.output/(name+'.portable_export.json'),read(portable/'completion.json'))
        rows.append(dict(id=name,passed=all(checks.values()),checks=checks))
    model=a.private/'custom/run/agent';spec=read(model/'task_spec.json');selection=read(model/'final_selection.json')
    adapter=a.private/'custom/run/delivery_recovery/sa25_revalidated/prediction_adapter.json'
    adapted=load_adapter(adapter,cases/'custom.late.zip',model,spec,selection)
    x=load_prediction_features(cases/'custom.query.zip',spec['model_kind'],spec)
    q=load_prediction_features(cases/'custom.late.zip',spec['model_kind'],adapted)
    protected=read(a.live/'late_replay/frozen_files.json')
    checks=dict(features=np.array_equal(x,q),two_exports=len({sha(a.private/'late_replay'/tag/'predictions.csv') for tag in ('first','second')})==1,
        same_as_reference=sha(a.private/'late_replay/first/predictions.csv')==sha(a.private/'custom/delivery/predictions.csv'),
        protected_files=protected=={n:sha(n) for n in protected},two_receipts=len(list((adapter.parent/'deliveries').glob('*.json')))==2,
        original_goal_preserved=all(spec[k]==adapted[k] for k in ('metric_spec','input_schema','feature_shape','task_kind')))
    rows.append(dict(id='late_replay',passed=all(checks.values()),checks=checks))
    evidence=read(a.windows_live/'windows_interruption/owned_processes.json');root=a.windows_private/'windows_interruption'
    receipt=read(root/'delivery/prediction_manifest.json');selection=read(root/'run/first_delivery/agent/final_selection.json')
    checks=dict(all_created_processes_exited=not any(alive(pid) for pid in evidence['owned_pids']),
        owned_tree=(evidence['controller']['parent']==evidence['controller_launcher'] and
            evidence['trainer_launch']['parent']==evidence['controller']['pid'] and evidence['trainer']['parent']==evidence['trainer_launch']['pid'] and
            evidence['worker_launch']['parent']==evidence['trainer']['pid'] and evidence['worker']['parent']==evidence['worker_launch']['pid']),
        paused_before_fit=evidence['worker']['stage']=='before_worker_module_and_fit',partial_disclosed=receipt['delivery_status']=='partial' and selection['refit'] is False,
        early_replay=sha(root/'delivery/predictions.csv')==sha(root/'replay/predictions.csv')==receipt['prediction_sha256'])
    checks['separate_windows_batch']=read(a.windows_live/'completion.json')['status']=='passed'
    rows.append(dict(id='windows_interruption',passed=all(checks.values()),checks=checks))
    historical=ledger_rows(ledger)==old;source=protocol['source_sha256']==identity()
    result=dict(status='passed' if all(r['passed'] for r in rows) and historical and source else 'failed',cases=rows,
        entire_canonical_ledger_unchanged=historical,source_unchanged=source,actual_classifier_fits=0,new_campus_API_requests=0,
        new_course_cost_rmb=0,ledger_now=ledger.project_summary(),scope='same A developer independent recheck, not human blind evaluation')
    result['evidence_sources']=dict(first_three='SA25B, original fourth fixture failure preserved',windows='SA25C separate zero-fit continuation')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
