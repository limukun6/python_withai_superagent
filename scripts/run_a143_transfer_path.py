"""One frozen real-task autonomous path; shared registered rows and persistent accounting."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from tools.ab_registered_run import run as registered_run
from tools.accepted_usage_snapshot import accepted_history
from scripts.campus_direct import campus_resolution,CAMPUS_IP

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--arm',choices=['L0','EC0'],required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('ledger','authorization','api-key-file'):p.add_argument('--'+n,type=Path)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();source=hashes(ROOT);registration=read(a.prepared/'results.json')
    paid=a.arm=='EC0';ledger=None;before=None
    if paid:
        from superagent.task_budget import TaskBudgetLedger
        ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary()
    save(out/'protocol.json',dict(batch='A143-'+a.arm,source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A142_A144_new_numeric_transfer_plan_20261124_zh.md'),
         prepared_completion_sha256=sha(a.prepared/'completion.json'),maximum_fits=0 if a.validate_only else 11,
         maximum_requests=0 if a.validate_only or not paid else 3,project_before=before,
         comparison='Whole local versus checked-compact configuration; not isolated compression effect'))
    def finish_owner():
        if ledger:
            save(out/'current_ledger_owner.json',dict(owner='A',window_status='exclusive_idle',ledger_path=str(a.ledger.resolve()),
                 ledger_sha256=sha(a.ledger),project_summary=ledger.project_summary(),active_paid_queue=False,
                 accepted_history_sha256=sha(a.authorization),existing_unknowns_accepted_by_user=True,new_unknowns_require_stop=True))
    try:
        if source!=registration['source_files']:raise ValueError('Pinned transfer source differs')
        for name,field in [('anonymous_train.zip','training_archive_sha256'),('registered_rows.json','registered_rows_sha256')]:
            if sha(a.prepared/name)!=registration[field]:raise ValueError('Registered transfer input changed')
        if paid:
            from superagent.research_budget import research_preflight
            if not a.api_key_file.is_file():raise FileNotFoundError('Configured local key missing')
            with accepted_history(a.authorization,sha(a.authorization)):
                research_preflight(ledger,'banknote_authentication',.30)
        if a.validate_only:
            save(out/'completion.json',dict(status='validated',actual_classifier_fits=0,real_API_requests=0));return
        command=['run',str(a.prepared/'anonymous_train.zip'),'--run-dir',str(out/'run'),
            '--planner','checked-history' if paid else 'local','--scheduler','ab-fixed','--portfolio','ab-v1',
            '--catalog-scope','full','--preference-catalog','standard','--parameter-semantics','off',
            '--fit-limit','11','--seed','707','--trial-timeout','300','--time-limit','1200','--partial-delivery','checkpoint']
        if paid:
            command+=['--feedback-view','compact','--proposal-contract','calibrated-v2',
                 '--history-policy','incremental-research','--api-key-file',str(a.api_key_file),
                 '--budget-ledger',str(a.ledger),'--task-budget-id','banknote_authentication','--llm-budget-rmb','.30']
            save(out/'current_ledger_owner.json',dict(owner='A',window_status='exclusive_active',ledger_path=str(a.ledger.resolve()),
                 project_summary=ledger.project_summary(),active_paid_queue=True,accepted_history_sha256=sha(a.authorization)))
        save(out/'command.json',command)
        auth=accepted_history(a.authorization,sha(a.authorization)) if paid else nullcontext()
        network=campus_resolution(CAMPUS_IP) if paid else nullcontext()
        with auth,network:code=registered_run(a.prepared/'registered_rows.json',registration['registered_rows_sha256'],command)
        outcome=read(out/'run/run_outcome.json');counts=outcome['classifier_fit_accounting']
        new_calls=ledger.project_summary()['requests']-before['requests'] if paid else 0
        if code or not outcome['requested_planner_completed'] or not 2<=counts['attempted_classifier_fits']<=11:
            raise RuntimeError('Autonomous path stopped without retry; retain its original outcome/checkpoint')
        if counts['attempted_classifier_fits']!=counts['completed_classifier_fits'] or counts['unknown_failed_attempts']:
            raise ValueError('Supervised worker accounting not complete')
        if outcome['new_api_requests']!=new_calls or new_calls>3:raise ValueError('Shared ledger/API counts differ')
        if paid:
            with accepted_history(a.authorization,sha(a.authorization)):research_preflight(ledger,'banknote_authentication',.30)
        manifest=read(out/'run/run_manifest.json')
        if manifest['capabilities']!=registration['capabilities']:raise ValueError('Common candidate catalogue changed')
        if manifest['source_files']!={k.removeprefix('superagent/'):v for k,v in source.items()}:raise ValueError('Training source identity drift')
        expected=read(a.prepared/'registered_rows.json')['splits']
        for part,row in zip(('primary','confirmation_1','confirmation_2'),expected):
            actual=read(out/'run'/(part+'_split.json'))
            for field in ('train_indices','search_indices','row_ids_sha256','label_sha256'):
                if actual[field]!=row[field]:raise ValueError('Actual internal row partition drift')
        selected=read(out/'run/final_selection.json')
        if sha(out/'run'/selected['model_file'])!=selected['model_sha256']:raise ValueError('Model seal differs')
        save(out/'model_frozen.json',dict(model_sha256=selected['model_sha256'],source_files=source,
             final_selection_sha256=sha(out/'run/final_selection.json'),outer_labels_scored=False))
        finish_owner();save(out/'completion.json',dict(status='sealed',actual_classifier_fits=counts['attempted_classifier_fits'],
             real_API_requests=new_calls,outer_performance_computed=False,elapsed_seconds=time.monotonic()-began,
             project_after=ledger.project_summary() if paid else None,source_unchanged=source==hashes(ROOT)))
    except Exception as e:
        finish_owner();run=out/'run/run_outcome.json';outcome=read(run) if run.is_file() else {}
        save(out/'completion.json',dict(status='stopped',error=str(e),automatic_retry=False,
             actual_classifier_fits=outcome.get('classifier_fit_accounting',{}).get('attempted_classifier_fits',0),
             real_API_requests=ledger.project_summary()['requests']-before['requests'] if paid else 0,
             elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':main()
