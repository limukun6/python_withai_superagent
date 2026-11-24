"""Unknown imported history stays unknown; new failures still stop spending."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from superagent.task_budget import TaskBudgetLedger,BudgetStateError
from superagent.research_budget import IncrementalResearchAccount,research_preflight


class IncrementalResearchTests(unittest.TestCase):
    def ledger(self,root):
        path=root/'history.json';path.write_text(json.dumps(dict(calls=[dict(request_id='d'*64,
            scope='shared_diagnostics',cost_status='unknown',course_cost_units=None,course_cost_rmb=None)])),encoding='utf-8')
        return TaskBudgetLedger.initialize(root/'ledger.sqlite',history=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_strict_stays_blocked_and_research_does_not_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(Path(tmp))
            with self.assertRaises(BudgetStateError):ledger.preflight('sms_spam',.3)
            state=research_preflight(ledger,'sms_spam',.3)
            self.assertFalse(state['accounting_complete']);self.assertFalse(state['valid_for_budgeted_final_report'])
            self.assertEqual(ledger.project_summary()['unresolved_requests'],1)
            self.assertIsNone(ledger.project_summary()['complete_course_cost_rmb'])

    def test_reserve_settle_and_new_unknown_or_pending_block_next_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(Path(tmp));account=IncrementalResearchAccount(ledger,'sms_spam','fixture')
            ticket=account.reserve(.1,'DeepSeek-V4.1-Flash')
            with self.assertRaises(BudgetStateError):research_preflight(ledger,'sms_spam',.1)
            account.settle(ticket,'DeepSeek-V4.1-Flash',100,20)
            self.assertGreater(ledger.project_summary()['known_course_cost_rmb'],0)
            self.assertEqual(ledger.project_summary()['unresolved_requests'],1)
            second=account.reserve(.1,'DeepSeek-V4.1-Flash');account.unknown(second,'controlled_failure')
            with self.assertRaisesRegex(BudgetStateError,'New unresolved request'):
                account.reserve(.1,'DeepSeek-V4.1-Flash')
            self.assertEqual(ledger.project_summary()['unresolved_requests'],2)

    def test_known_ceiling_is_still_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(Path(tmp))
            with self.assertRaises(BudgetStateError):research_preflight(ledger,'sms_spam',20.01)

    def test_formal_cli_preflight_requires_explicit_research_mode(self):
        from unittest.mock import patch
        from superagent.ab_flash import preflight
        from superagent.cli import build_parser
        import os
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}):
            ledger=self.ledger(Path(tmp))
            flags=['run','missing.zip','--scheduler','ab-v1','--portfolio','ab-v1','--planner','raw-history',
                '--budget-ledger',str(ledger.path),'--task-budget-id','sms_spam','--api-key-file','missing']
            with self.assertRaises(ValueError):preflight(build_parser().parse_args(flags))
            _,_,state=preflight(build_parser().parse_args(flags+['--history-policy','incremental-research']))
            self.assertEqual(state['course_budget_compliance'],'unverified_history')


if __name__=='__main__':unittest.main()
