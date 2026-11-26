"""Explicit continuation against unchanged, user-accepted unknown ledger rows.

These are integrity-bound research permissions, not provider bill reconciliation.
The production account checks each reservation in the real ledger transaction.
"""
import hashlib
import json
import math
from pathlib import Path
import uuid

from .task_budget import BudgetStateError, encoded, now
from .usage_audit import money_units


def row_digest(row):
    return hashlib.sha256(encoded(dict(row)).encode('utf-8')).hexdigest()


def load_snapshot(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise BudgetStateError('Accepted-history snapshot SHA mismatch')
    record = json.loads(raw)
    if record.get('version') != 1 or record.get('policy') != 'user_accepted_existing_unknowns':
        raise BudgetStateError('Unsupported accepted-history policy')
    if not record.get('user_authorization') or not record.get('accepted_unknown_requests'):
        raise BudgetStateError('Explicit user authorization and enumerated request identities required')
    entries = record['accepted_unknown_requests']
    if len({e['id'] for e in entries}) != len(entries):
        raise BudgetStateError('Duplicate accepted request IDs')
    return record


def check_snapshot(db, state, amount, record):
    database = next((r[2] for r in db.execute('PRAGMA database_list') if r[1] == 'main'), None)
    if Path(database or '').resolve() != Path(record['ledger_path']).resolve():
        raise BudgetStateError('Accepted history belongs to a different real ledger')
    accepted = {row['id']: row['row_sha256'] for row in record['accepted_unknown_requests']}
    observed = set()
    for row in db.execute("SELECT * FROM requests WHERE status IN ('unknown','reserved')"):
        if row['status'] == 'reserved':
            raise BudgetStateError('Pending reservation blocks accepted-history continuation')
        if row['id'] not in accepted or row_digest(row) != accepted[row['id']]:
            raise BudgetStateError('New or modified unknown request blocks accepted-history continuation')
        observed.add(row['id'])
    if observed != set(accepted):
        raise BudgetStateError('Accepted unknown history changed; explicit revalidation required')
    value = money_units(amount)
    known = money_units(state['known_cost_counted_against_limit_rmb'])
    if value <= 0 or known + value > 20_000_000_000:
        raise BudgetStateError('Known costs plus incremental allowance exceed course limit')
    return value


class AcceptedHistoryAccount:
    def __init__(self, ledger, scope, run_id, snapshot_path, snapshot_sha256):
        self.ledger, self.scope, self.run_id = ledger, scope, run_id
        self.record = load_snapshot(snapshot_path, snapshot_sha256)
        self.snapshot_sha256 = snapshot_sha256
        if self.record.get('allowed_scope') != scope:
            raise BudgetStateError('Accepted-history scope mismatch')
        count = self.record.get('maximum_new_requests')
        limit = self.record.get('estimated_stop_threshold_rmb')
        if type(count) is not int or count < 1 or type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= 20:
            raise BudgetStateError('Accepted-history request count and estimated stop threshold required')
        self.max_requests, self.allowance = count, float(limit)

    def _check(self, db, amount):
        state = self.ledger._snapshot(db, self.scope)
        units = check_snapshot(db, state, amount, self.record)
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        if metadata.get('origin_sha256') in (None, 'explicit_new_project') or not db.execute('SELECT 1 FROM requests LIMIT 1').fetchone():
            raise BudgetStateError('Accepted history requires nonempty imported cumulative usage')
        rows = db.execute('SELECT course_units,reserved_units FROM requests WHERE scope=?', (self.scope,)).fetchall()
        if len(rows) >= self.max_requests:
            raise BudgetStateError('Accepted-history cumulative request limit reached')
        if sum(r['course_units'] or r['reserved_units'] or 0 for r in rows) + units > money_units(self.allowance):
            raise BudgetStateError('Accepted-history cumulative estimated stop threshold reached')
        return state, units

    def annotate(self, state):
        return {**state, 'history_policy': 'user-accepted-snapshot-research',
                'course_budget_compliance': 'unverified_history', 'valid_for_budgeted_final_report': False,
                'accepted_history_sha256': self.snapshot_sha256,
                'accepted_unknown_request_count': len(self.record['accepted_unknown_requests']),
                'policy_note': 'Enumerated old charges remain unknown/null; new unknown/pending blocks. Provider total is unverified.'}

    def preflight(self):
        with self.ledger._db() as db:
            # A nominal reservation checks remaining capacity without claiming a hard cost bound.
            state, _ = self._check(db, .000000001)
        return self.annotate(state)

    def reserve(self, amount, model):
        with self.ledger._db() as db:
            db.execute('BEGIN IMMEDIATE')
            _, units = self._check(db, amount)
            key = str(uuid.uuid4())
            payload = encoded(dict(run_id=str(self.run_id), requested_model=model, reserved_at=now(),
                                   history_policy='user-accepted-snapshot-research',
                                   accepted_history_sha256=self.snapshot_sha256,
                                   course_budget_compliance='unverified_history', estimate_is_not_a_hard_token_bound=True))
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)', (key, self.scope, 'reserved', units, None, payload))
            db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)', (key, 'reserved', now(), payload))
            db.commit()
        return key

    def settle(self, key, model, inp, out):
        return self.annotate(self.ledger.settle(key, model, inp, out))

    def unknown(self, key, reason, **details):
        return self.ledger.mark_unknown(key, reason, **details)

    def snapshot(self):
        return self.annotate(self.ledger.snapshot(self.scope))
