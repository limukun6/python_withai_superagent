"""Explicit research continuation with unreconciled IMPORTED history kept unknown.

This account does not certify the full course budget. Any new unknown request
or pending reservation blocks further calls. No historical row is settled as zero.
"""
import uuid

from superagent.task_budget import BudgetStateError,encoded,now
from superagent.usage_audit import money_units


def research_preflight(ledger,scope,amount):
    with ledger._db() as db:
        state=ledger._snapshot(db,scope)
        _check(db,state,amount)
    return annotate(state)


def _check(db,state,amount):
    rows=db.execute("SELECT id,status FROM requests WHERE status IN ('unknown','reserved')").fetchall()
    if any(r['status']=='reserved' or not r['id'].startswith('historical:') for r in rows):
        raise BudgetStateError('New unresolved request or pending reservation blocks research continuation')
    value=money_units(amount)
    known=money_units(state['known_cost_counted_against_limit_rmb'])
    if value<=0 or known+value>20_000_000_000:
        raise BudgetStateError('Known costs plus incremental allowance exceed the course limit')
    return value


def annotate(state):
    return {**state,'history_policy':'incremental-research','course_budget_compliance':'unverified_history',
        'valid_for_budgeted_final_report':False,
        'policy_note':'Known charges limited; imported historical unknown charges retain null cost and have no certified upper bound'}


class IncrementalResearchAccount:
    def __init__(self,ledger,scope,run_id):self.ledger,self.scope,self.run_id=ledger,scope,run_id
    def reserve(self,amount,model):
        with self.ledger._db() as db:
            db.execute('BEGIN IMMEDIATE');state=self.ledger._snapshot(db,self.scope);units=_check(db,state,amount)
            key=str(uuid.uuid4());payload=dict(run_id=str(self.run_id),requested_model=model,reserved_at=now(),
                history_policy='incremental-research',course_budget_compliance='unverified_history',
                estimate_is_not_a_hard_token_bound=True)
            value=encoded(payload)
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',(key,self.scope,'reserved',units,None,value))
            db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',
                (key,'reserved',now(),value));db.commit();return key
    def settle(self,key,model,inp,out):return annotate(self.ledger.settle(key,model,inp,out))
    def unknown(self,key,reason,**details):return self.ledger.mark_unknown(key,reason,**details)
    def snapshot(self):return annotate(self.ledger.snapshot(self.scope))
