"""Persistent course-cost accounting for the documented CLI, using SQLite transactions."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .pricing import resolve_pricing
from .usage_audit import money_units

SHARED='shared_diagnostics'
LIMIT=20_000_000_000


class BudgetStateError(ValueError):pass


def now():return datetime.now(timezone.utc).isoformat()


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


class TaskBudgetLedger:
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve()
        if not self.path.is_file():raise BudgetStateError('Budget ledger is not initialized; import verified history or explicitly initialize a new project')
        with self._db() as db:
            try:version=db.execute("SELECT value FROM metadata WHERE key='version'").fetchone()
            except sqlite3.DatabaseError:raise BudgetStateError('Not an initialized SuperAgent budget ledger') from None
            if version is None or version[0]!='1':raise BudgetStateError('Unsupported budget ledger version')

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        try:yield db
        except BaseException:
            db.rollback();raise
        finally:db.close()

    @classmethod
    def initialize(cls,path,history=None,expected_sha256=None,new_project=False):
        target=Path(path).expanduser().resolve()
        if bool(history)==bool(new_project):raise BudgetStateError('Choose verified history import or explicit new-project initialization')
        records=[]
        if history:
            source=Path(history);digest=hashlib.sha256(source.read_bytes()).hexdigest()
            if not expected_sha256 or digest!=expected_sha256:raise BudgetStateError('Historical audit SHA256 does not match')
            audit=json.loads(source.read_text());records=cls._historical_records(audit)
        else:digest=None
        target.parent.mkdir(parents=True,exist_ok=True)
        # Atomic creation prevents accidentally overwriting or reinitializing an old ledger.
        descriptor=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
        try:
            db=sqlite3.connect(target)
            with db:
                db.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE requests(id TEXT PRIMARY KEY,scope TEXT NOT NULL,status TEXT NOT NULL,
    reserved_units INTEGER NOT NULL,course_units INTEGER,payload TEXT NOT NULL);
CREATE TABLE transitions(seq INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT NOT NULL,
    event TEXT NOT NULL,time_utc TEXT NOT NULL,payload TEXT NOT NULL);''')
                db.execute('INSERT INTO metadata VALUES (?,?)',('version','1'))
                db.execute('INSERT INTO metadata VALUES (?,?)',('initialized_at',now()))
                db.execute('INSERT INTO metadata VALUES (?,?)',('origin_sha256',digest or 'explicit_new_project'))
                cls._import_rows(db,records)
            db.close()
        except BaseException:
            try:db.close()
            except Exception:pass
            target.unlink(missing_ok=True);raise
        return cls(target)

    @staticmethod
    def _historical_records(audit):
        rows=audit.get('calls')
        if not isinstance(rows,list):raise BudgetStateError('Historical audit needs an explicit calls list')
        result=[]
        for row in rows:
            key=row.get('request_id');scope=row.get('scope')
            if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
                raise BudgetStateError('Historical request has no verified event fingerprint')
            if not isinstance(scope,str) or not scope.strip() or scope!=scope.strip().casefold():
                raise BudgetStateError('Historical scope must be a stable lowercase identifier')
            status=row.get('cost_status');cost=row.get('course_cost_units')
            if status=='known':
                price=resolve_pricing(row.get('served_model') or row.get('requested_model'))
                expected=money_units(price.course_cost(row.get('input_tokens'),row.get('output_tokens')))
                if type(cost) is not int or cost!=expected or money_units(row.get('course_cost_rmb'))!=expected:
                    raise BudgetStateError('Historical tokens and course cost conflict')
                if (row.get('pricing') or {}).get('price_version')!=price.price_version:
                    raise BudgetStateError('Historical price version is unverified')
            elif status=='unknown':
                if cost is not None or row.get('course_cost_rmb') is not None:raise BudgetStateError('Unknown history cannot be imported as zero')
            else:raise BudgetStateError('Unsupported historical accounting state')
            result.append(('historical:'+key,scope,status,0,cost,encoded(row)))
        return result

    @staticmethod
    def _import_rows(db,rows):
        for values in rows:
            prior=db.execute('SELECT id,scope,status,reserved_units,course_units,payload FROM requests WHERE id=?',(values[0],)).fetchone()
            if prior is not None:
                if tuple(prior)!=values:raise BudgetStateError('Conflicting historical import; existing call is immutable')
                continue
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',values)
            db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',
                (values[0],'historical_import',now(),values[-1]))

    def import_history(self,path,expected_sha256):
        source=Path(path)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected_sha256:raise BudgetStateError('Historical audit SHA256 does not match')
        rows=self._historical_records(json.loads(source.read_text()))
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE');self._import_rows(db,rows);db.commit()

    @staticmethod
    def _scope(scope):
        if not isinstance(scope,str) or not scope.strip() or scope==SHARED or scope!=scope.strip().casefold():
            raise BudgetStateError('Use a stable lowercase task budget ID, not the shared diagnostic scope')

    def _snapshot(self,db,scope):
        self._scope(scope)
        rows=db.execute('SELECT scope,status,course_units,reserved_units FROM requests WHERE scope IN (?,?)',(scope,SHARED)).fetchall()
        own=sum(r['course_units'] or 0 for r in rows if r['scope']==scope)
        shared=sum(r['course_units'] or 0 for r in rows if r['scope']==SHARED)
        unknown=sum(r['status']=='unknown' for r in rows);pending=sum(r['status']=='reserved' for r in rows)
        complete=unknown==0 and pending==0
        return {'task_budget_id':scope,'task_limit_rmb':20.,'known_task_cost_rmb':own/1e9,
            'shared_known_cost_rmb':shared/1e9,'known_cost_counted_against_limit_rmb':(own+shared)/1e9,
            'unresolved_requests':unknown,'pending_requests':pending,'accounting_complete':complete,
            'remaining_course_allowance_rmb':max(0,LIMIT-own-shared)/1e9 if complete else None,
            'limit_exceeded':own+shared>LIMIT,'actual_account_debit_rmb':None,
            'shared_cost_rule':'Fully reserve shared diagnostics against each task limit; count once in the project total'}

    def snapshot(self,scope):
        with self._db() as db:return self._snapshot(db,scope)

    @staticmethod
    def _check(state,amount):
        units=money_units(amount)
        if units<=0:raise BudgetStateError('Reservation/allowance must be positive')
        if not state['accounting_complete']:
            raise BudgetStateError('Task or shared scope has unresolved/pending requests; reconcile usage before another API call')
        if units>money_units(state['remaining_course_allowance_rmb']):
            raise BudgetStateError('Cumulative task course allowance is insufficient')
        return units

    def _require_fresh_outbox_scope(self,db,scope):
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        prior = db.execute('SELECT 1 FROM requests LIMIT 1').fetchone()
        conflict = db.execute("SELECT 1 FROM requests WHERE status != 'known' OR scope = ? LIMIT 1",(scope,)).fetchone()
        if (metadata.get('ledger_kind') != 'provisional_local_outbox'
                or metadata.get('team_merge_state') != 'pending_team_handoff' or prior is None or conflict):
            raise BudgetStateError('Unsettled, repeated, empty or incorrectly marked research outbox')

    def preflight(self,scope,run_allowance,*,fresh_outbox_scope=False):
        with self._db() as db:
            state = self._snapshot(db,scope)
            self._check(state,run_allowance)
            if fresh_outbox_scope:
                self._require_fresh_outbox_scope(db,scope)
            return state

    def reserve(self,scope,amount,run_id,model,*,fresh_outbox_scope=False):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE');units=self._check(self._snapshot(db,scope),amount)
            if fresh_outbox_scope:
                self._require_fresh_outbox_scope(db,scope)
            key=str(uuid.uuid4());payload={'run_id':str(run_id),'requested_model':model,'reserved_at':now(),'estimate_is_not_a_hard_token_bound':True}
            value=encoded(payload)
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',(key,scope,'reserved',units,None,value))
            db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',(key,'reserved',now(),value));db.commit()
            return key

    def _finish(self,key,status,cost,details):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT * FROM requests WHERE id=?',(key,)).fetchone()
            if row is None:raise BudgetStateError('Unknown request reservation')
            if row['status']!='reserved':raise BudgetStateError('A terminal accounting event cannot be overwritten')
            payload={**json.loads(row['payload']),'resolution':details,'resolved_at':now()}
            db.execute('UPDATE requests SET status=?,course_units=?,payload=? WHERE id=?',(status,cost,encoded(payload),key))
            db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',(key,status,now(),encoded(details)))
            state=self._snapshot(db,row['scope']);db.commit();return state

    def settle(self,key,model,input_tokens,output_tokens):
        price=resolve_pricing(model);cost=money_units(price.course_cost(input_tokens,output_tokens))
        return self._finish(key,'known',cost,{'model':model,'input_tokens':input_tokens,'output_tokens':output_tokens,
            'course_cost_rmb':cost/1e9,'pricing':price.metadata(),'actual_account_debit_rmb':None})

    def mark_unknown(self,key,reason,**known):
        return self._finish(key,'unknown',None,{'reason':reason,**known,'course_cost_rmb':None,'actual_account_debit_rmb':None})

    def project_summary(self):
        with self._db() as db:
            rows=db.execute('SELECT status,course_units FROM requests').fetchall()
        known=sum(r['course_units'] or 0 for r in rows);unknown=sum(r['status']=='unknown' for r in rows);pending=sum(r['status']=='reserved' for r in rows)
        return {'requests':len(rows),'known_course_cost_rmb':known/1e9,'unresolved_requests':unknown,'pending_requests':pending,
            'complete_course_cost_rmb':known/1e9 if not unknown and not pending else None,'actual_account_debit_rmb':None}


class TaskBudgetAccount:
    def __init__(self,ledger,scope,run_id,*,fresh_outbox_scope=False):
        self.ledger,self.scope,self.run_id=ledger,scope,run_id
        self.fresh_outbox_scope=fresh_outbox_scope

    def reserve(self,amount,model):
        if self.fresh_outbox_scope:
            return self.ledger.reserve(self.scope,amount,self.run_id,model,fresh_outbox_scope=True)
        return self.ledger.reserve(self.scope,amount,self.run_id,model)
    def settle(self,key,model,inp,out):return self.ledger.settle(key,model,inp,out)
    def unknown(self,key,reason,**details):return self.ledger.mark_unknown(key,reason,**details)
    def snapshot(self):return self.ledger.snapshot(self.scope)
