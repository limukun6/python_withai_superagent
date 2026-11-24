import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.accepted_usage_snapshot import check_snapshot,row_digest,accepted_history,load_snapshot
from superagent import research_budget
from superagent.task_budget import BudgetStateError


class AcceptedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'fixture.sqlite';self.db=sqlite3.connect(self.path);self.db.row_factory=sqlite3.Row
        self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE requests(id TEXT,scope TEXT,status TEXT,reserved_units INTEGER,course_units INTEGER,payload TEXT)')
        self.db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',('native-unknown','fixture','unknown',0,None,'{}'))
        row=self.db.execute('SELECT * FROM requests').fetchone()
        self.record=dict(version=1,policy='user_accepted_existing_unknowns',user_authorization='Test fixture only; no HTTP',
            ledger_path=str(self.path.resolve()),accepted_unknown_requests=[dict(id='native-unknown',row_sha256=row_digest(row))])
        self.state={'known_cost_counted_against_limit_rmb':0}

    def test_exact_accepted_unknown_is_admitted_without_settlement(self):
        self.assertGreater(check_snapshot(self.db,self.state,.3,self.record),0)
        self.assertIsNone(self.db.execute('SELECT course_units FROM requests').fetchone()[0])

    def test_new_unknown_is_rejected(self):
        self.db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',('new','fixture','unknown',0,None,'{}'))
        with self.assertRaisesRegex(BudgetStateError,'New or modified'):check_snapshot(self.db,self.state,.3,self.record)

    def test_pending_is_rejected_even_for_previously_named_id(self):
        self.db.execute("UPDATE requests SET status='reserved'")
        with self.assertRaisesRegex(BudgetStateError,'Pending'):check_snapshot(self.db,self.state,.3,self.record)

    def test_modified_unknown_record_is_rejected(self):
        self.db.execute("UPDATE requests SET payload='changed'")
        with self.assertRaisesRegex(BudgetStateError,'New or modified'):check_snapshot(self.db,self.state,.3,self.record)

    def test_wrong_ledger_is_rejected(self):
        self.record['ledger_path']=str(self.path.with_name('other.sqlite'))
        with self.assertRaisesRegex(BudgetStateError,'different real ledger'):check_snapshot(self.db,self.state,.3,self.record)

    def test_known_cost_limit_remains_enforced(self):
        with self.assertRaisesRegex(BudgetStateError,'course limit'):check_snapshot(self.db,{'known_cost_counted_against_limit_rmb':19.9},.3,self.record)

    def test_explicit_context_and_sha_required_and_original_policy_restored(self):
        with self.assertRaisesRegex(BudgetStateError,'unresolved'):research_budget._check(self.db,self.state,.3)
        path=self.path.with_suffix('.json');raw=json.dumps(self.record).encode();path.write_bytes(raw)
        digest=hashlib.sha256(raw).hexdigest();original=research_budget._check
        with self.assertRaisesRegex(BudgetStateError,'SHA'):load_snapshot(path,'wrong')
        with accepted_history(path,digest):
            self.assertGreater(research_budget._check(self.db,self.state,.3),0)
            self.assertFalse(research_budget.annotate(self.state)['valid_for_budgeted_final_report'])
        self.assertIs(research_budget._check,original)
        with self.assertRaises(BudgetStateError):original(self.db,self.state,.3)


if __name__=='__main__':unittest.main()
