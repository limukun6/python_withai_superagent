"""Freeze both models before opening a new task's held labels; at most 22 fits and 3 calls."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,child,physical

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','output','ledger','authorization','api-key-file'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();source=hashes(ROOT)
    from superagent.task_budget import TaskBudgetLedger
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary()
    save(out/'protocol.json',dict(batch='A143-queue',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A142_A144_new_numeric_transfer_plan_20261124_zh.md'),
         maximum_fits=0 if a.validate_only else 22,maximum_API_requests=0 if a.validate_only else 3,
         starting_A_fits=1614,project_before=before,order=['L0','EC0'],outer_labels_available_to_agent=False))
    records=[]
    try:
        private=out/'private';private.mkdir()
        with sqlite3.connect(a.ledger) as db,sqlite3.connect(private/'ledger_before.sqlite') as backup:db.backup(backup)
        for arm in ('L0','EC0'):
            command=[sys.executable,'-m','scripts.run_a143_transfer_path','--prepared',str(a.prepared),
                '--arm',arm,'--output',str(out/arm)]
            if arm=='EC0':command+=['--ledger',str(a.ledger),'--authorization',str(a.authorization),'--api-key-file',str(a.api_key_file)]
            if a.validate_only:command+=['--validate-only']
            env=environment(out);env['AB_A29_RUN_NAME']=arm;env['SUPERAGENT_OFFLINE']='0' if arm=='EC0' else '1'
            process=child(command,out/(arm+'.log'),env,1380,allow_failure=True)
            complete=read(out/arm/'completion.json');records.append(dict(arm=arm,process=process,completion=complete))
            save(out/'checkpoint.json',records)
            if process['returncode'] or complete['status']!=('validated' if a.validate_only else 'sealed'):
                raise RuntimeError('Transfer queue stopped without retry: '+arm)
            traced=[r for r in physical(out) if r['run_name']==arm]
            if len(traced)!=complete['actual_classifier_fits']:raise ValueError('Observed physical fits differ from formal ledger')
            print(json.dumps(dict(arm=arm,status=complete['status'],fits=len(traced),new_API=complete['real_API_requests'])),flush=True)
        after=ledger.project_summary()
        if len(physical(out))>22 or after['requests']-before['requests']>3 or hashes(ROOT)!=source:raise ValueError('Budget/source drift')
        save(out/'results.json',dict(paths=records,project_before=before,project_after=after,
             known_incremental_cost_rmb=after['known_course_cost_rmb']-before['known_course_cost_rmb']))
        save(out/'completion.json',dict(status='validated' if a.validate_only else 'all_models_sealed',
             actual_classifier_fits=len(physical(out)),real_API_requests=after['requests']-before['requests'],
             A_cumulative_fits_after=1614+len(physical(out)),outer_labels_scored=False,source_unchanged=True,
             elapsed_seconds=time.monotonic()-began,protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    except Exception as e:
        after=ledger.project_summary();save(out/'completion.json',dict(status='stopped',error=str(e),
             actual_classifier_fits=len(physical(out)),real_API_requests=after['requests']-before['requests'],
             A_cumulative_fits_after=1614+len(physical(out)),automatic_retry=False,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':main()
