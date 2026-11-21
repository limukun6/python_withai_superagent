"""Do not mistake a timeout or a mismatched request for an unsent DNS attempt."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.resolve_a25_no_dispatch import resolve
from superagent.task_budget import TaskBudgetLedger
from superagent.research_budget import IncrementalResearchAccount


class NoDispatchTests(unittest.TestCase):
    def fixture(self,root,error='ConnectionError: NameResolutionError getaddrinfo failed',wrong_ticket=False,http=False):
        audit=root/'audit.json';audit.write_text(json.dumps(dict(calls=[dict(request_id='e'*64,scope='shared_diagnostics',
            cost_status='unknown',course_cost_units=None,course_cost_rmb=None)])),encoding='utf-8')
        ledger=TaskBudgetLedger.initialize(root/'ledger.sqlite',history=audit,
            expected_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
        run=root/'run';run.mkdir();account=IncrementalResearchAccount(ledger,'fixture',str(run.resolve()))
        ticket=account.reserve(.1,'DeepSeek-V4.1-Flash');account.unknown(ticket,'controlled fixture')
        usage=dict(event='unresolved_call',role='planner',budget_request_id=ticket)
        events=[dict(kind='llm_request',role='planner',budget_request_id='wrong' if wrong_ticket else ticket),
                dict(kind='llm_transport_failure',role='planner',error=error,http_status=None,data_events=0)]
        if http:events.append(dict(kind='llm_response',role='planner',content='something'))
        (run/'llm_usage.jsonl').write_text(json.dumps(usage)+'\n',encoding='utf-8')
        (run/'events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n',encoding='utf-8')
        return ledger,run,ticket

    def test_verified_dns_appends_a_review_and_keeps_the_original_unknown_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger,run,ticket=self.fixture(root);resolve(ledger,run,root/'review')
            self.assertEqual(ledger.project_summary()['unresolved_requests'],1)
            with ledger._db() as db:
                row=db.execute('SELECT status,course_units FROM requests WHERE id=?',(ticket,)).fetchone()
                events=[r[0] for r in db.execute('SELECT event FROM transitions WHERE request_id=?',(ticket,))]
            self.assertEqual(tuple(row),('not_dispatched',0))
            self.assertIn('unknown',events);self.assertIn('reviewed_not_dispatched',events)

    def test_timeout_response_and_wrong_ticket_cannot_be_zeroed(self):
        cases=[dict(error='ReadTimeout'),dict(wrong_ticket=True),dict(http=True)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ledger,run,_=self.fixture(root,**kwargs)
                with self.assertRaises(ValueError):resolve(ledger,run,root/'review')
                self.assertEqual(ledger.project_summary()['unresolved_requests'],2)


if __name__=='__main__':unittest.main()
