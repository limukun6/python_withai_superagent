"""Explicit user-accepted unresolved-history snapshot, scoped to one real ledger and context."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from superagent.task_budget import BudgetStateError,encoded
from superagent.usage_audit import money_units


def row_digest(row):
    return hashlib.sha256(encoded(dict(row)).encode('utf-8')).hexdigest()


def load_snapshot(path,expected_sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise BudgetStateError('Accepted-history snapshot SHA mismatch')
    record=json.loads(raw)
    if record.get('version')!=1 or record.get('policy')!='user_accepted_existing_unknowns':
        raise BudgetStateError('Unsupported accepted-history policy')
    if not record.get('user_authorization') or not record.get('accepted_unknown_requests'):
        raise BudgetStateError('Explicit user authorization and enumerated request identities required')
    entries=record['accepted_unknown_requests']
    if len({e['id'] for e in entries})!=len(entries):raise BudgetStateError('Duplicate accepted request IDs')
    return record


def check_snapshot(db,state,amount,record):
    database=next((r[2] for r in db.execute('PRAGMA database_list') if r[1]=='main'),None)
    if str(Path(database or '').resolve())!=str(Path(record['ledger_path']).resolve()):
        raise BudgetStateError('Accepted history belongs to a different real ledger')
    accepted={row['id']:row['row_sha256'] for row in record['accepted_unknown_requests']}
    for row in db.execute("SELECT * FROM requests WHERE status IN ('unknown','reserved')"):
        if row['status']=='reserved':raise BudgetStateError('Pending reservation blocks accepted-history continuation')
        if row['id'] not in accepted or row_digest(row)!=accepted[row['id']]:
            raise BudgetStateError('New or modified unknown request blocks accepted-history continuation')
    value=money_units(amount);known=money_units(state['known_cost_counted_against_limit_rmb'])
    if value<=0 or known+value>20_000_000_000:
        raise BudgetStateError('Known costs plus incremental allowance exceed course limit')
    return value


@contextmanager
def accepted_history(path,expected_sha256):
    """Keep all unknown costs null; only the pre-existing named rows are admitted.

    The production planner/executor stays byte-identical to frozen L0 models.
    Checks run inside the ledger's normal transactions, including each reserve.
    """
    from superagent import research_budget
    record=load_snapshot(path,expected_sha256);original_annotate=research_budget.annotate
    def check(db,state,amount):return check_snapshot(db,state,amount,record)
    def annotate(state):
        return {**original_annotate(state),'history_policy':'user-accepted-snapshot-research',
            'accepted_history_sha256':expected_sha256,'accepted_unknown_request_count':len(record['accepted_unknown_requests']),
            'policy_note':'User accepts enumerated existing unknown charges without settlement. Costs stay null; any new unknown/pending blocks. Known-cost limit remains enforced.'}
    with patch.object(research_budget,'_check',check),patch.object(research_budget,'annotate',annotate):
        yield record
