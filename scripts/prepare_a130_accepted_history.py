"""Record the user's current unknown-history authorization; never settle or reset old requests."""
import argparse
from pathlib import Path
import sqlite3
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from tools.accepted_usage_snapshot import row_digest,accepted_history

ROOT=Path(__file__).resolve().parents[1]
USER_AUTHORIZATION='付费不要暂停，我们没法调到官网账单，不管他了'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A130-authorize',ledger_before_sha256=sha(a.ledger),
        runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A130_A132_H1_paid_continuation_plan_20261123_zh.md'),
        user_authorization=USER_AUTHORIZATION,maximum_classifier_fits=0,real_API_requests=0))
    try:
        from superagent.task_budget import TaskBudgetLedger
        from superagent.research_budget import research_preflight
        ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();original=sha(a.ledger)
        if (before['requests'],before['unresolved_requests'],before['pending_requests'])!=(357,12,0):
            raise ValueError('Current ledger no longer matches the explicitly accepted 357/12 snapshot')
        with ledger._db() as db:rows=[dict(r) for r in db.execute("SELECT * FROM requests WHERE status='unknown' ORDER BY id")]
        snapshot=dict(version=1,policy='user_accepted_existing_unknowns',user_authorization=USER_AUTHORIZATION,
            ledger_path=str(a.ledger.resolve()),ledger_before_sha256=original,project_summary_before=before,
            accepted_unknown_requests=[dict(id=r['id'],scope=r['scope'],row_sha256=row_digest(r)) for r in rows],
            unknown_costs_remain_null=True,new_unknowns_or_pending_block=True,course_budget_compliance='unverified_history')
        target=a.output/'accepted_history.json';save(target,snapshot);digest=sha(target)
        with accepted_history(target,digest):state=research_preflight(ledger,'h1_transfer',.30)
        # Make an ignored audit backup; the same received ledger remains the only writer target.
        private=a.output/'private';private.mkdir()
        with sqlite3.connect(a.ledger) as source,sqlite3.connect(private/'before.sqlite') as backup:source.backup(backup)
        if sha(a.ledger)!=original:raise ValueError('Preflight changed original history')
        owner=dict(owner='A',window_status='exclusive_active',ledger_sha256=original,project_summary=before,
            accepted_history_sha256=digest,authority='User continuation instruction; B had stopped writing',
            history_unresolved=True,new_paid_calls_permitted=True)
        save(a.output/'ledger_owner.json',owner)
        save(a.output/'results.json',dict(accepted_unknown_requests=12,accepted_history_sha256=digest,
            original_ledger_unchanged=True,state=state,all_old_costs_unchanged=True,owner='A'))
        save(a.output/'completion.json',dict(status='authorized_and_validated',actual_classifier_fits=0,real_API_requests=0,
            elapsed_seconds=time.monotonic()-start))
        print(dict(status='authorized',unknowns_preserved=12,ledger_unchanged=True,API=0),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
