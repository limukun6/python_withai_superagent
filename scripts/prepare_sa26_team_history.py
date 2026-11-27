"""Bind standing user authorization to the exact existing real ledger on the receiving machine."""
import argparse
from pathlib import Path

from scripts.run_sa5_semantic_panel import save,sha
from superagent.accepted_history import row_digest,AcceptedHistoryAccount
from superagent.task_budget import TaskBudgetLedger


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary()
    if not before['requests'] or before['pending_requests']:raise ValueError('Need nonempty real history and no pending requests')
    with ledger._db() as db:
        all_before={r['id']:row_digest(r) for r in db.execute('SELECT * FROM requests')}
        unknown=[dict(r) for r in db.execute("SELECT * FROM requests WHERE status='unknown' ORDER BY id")]
    if not unknown:raise ValueError('No unknown history: use the ordinary strict account instead')
    a.output.mkdir(parents=True,exist_ok=False)
    record=dict(version=1,policy='user_accepted_existing_unknowns',
        user_authorization='2026-11-27 user: ignore unknown usage and continue bounded paid research; user now delegates remaining work to teammate. Failure still ends current batch with no HTTP retry.',
        ledger_path=str(a.ledger.resolve()),ledger_before_sha256=sha(a.ledger),project_summary_before=before,
        allowed_scope='sa26_handoff_parent',maximum_new_requests=3,estimated_stop_threshold_rmb=.30,
        accepted_unknown_requests=[dict(id=r['id'],scope=r['scope'],row_sha256=row_digest(r)) for r in unknown],
        unknown_costs_remain_null=True,course_budget_compliance='unverified_history',
        use='Parent profile only. SA26 runner clones a fresh registered per-case scope; never reuse a paid scope or retry a failed request.')
    path=a.output/'accepted_history.json';save(path,record)
    account=AcceptedHistoryAccount(ledger,'sa26_handoff_parent',str(a.output),path,sha(path));account.preflight()
    with ledger._db() as db:unchanged=all_before=={r['id']:row_digest(r) for r in db.execute('SELECT * FROM requests')}
    if not unchanged:raise ValueError('Ledger changed during snapshot creation')
    save(a.output/'completion.json',dict(status='prepared',unknown_rows=len(unknown),requests=before['requests'],
        original_rows_unchanged=True,accepted_history_sha256=sha(path),actual_classifier_fits=0,campus_API_requests=0))
    print(dict(status='prepared',requests=before['requests'],unknown_rows=len(unknown),fits=0,API=0))


if __name__=='__main__':main()
