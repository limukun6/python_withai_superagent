"""Production research permissions and campus transport: no real API or fitting."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from superagent.accepted_history import AcceptedHistoryAccount, row_digest
from superagent.campus_transport import CampusAdapter, CampusConnection, CAMPUS_IP, ENDPOINT, HOST
from superagent.events import RunLogger
from superagent.task_budget import BudgetStateError, TaskBudgetLedger
from superagent.unified import configured_flash
from tests.test_task_budget import history_row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LiveEntryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        history = self.root/'history.json'
        history.write_text(json.dumps({'calls': [history_row('old','old_scope')]}))
        self.ledger = TaskBudgetLedger.initialize(self.root/'ledger.sqlite', history=history, expected_sha256=sha(history))
        with self.ledger._db() as db:
            old = db.execute('SELECT * FROM requests').fetchone()
        self.old = dict(old)
        self.record = dict(version=1, policy='user_accepted_existing_unknowns', user_authorization='synthetic test fixture',
                           ledger_path=str(self.ledger.path), allowed_scope='new_scope', maximum_new_requests=2,
                           estimated_stop_threshold_rmb=.25,
                           accepted_unknown_requests=[dict(id=old['id'],row_sha256=row_digest(old))])
        self.path = self.root/'accepted.json'; self.write()
        self.key = self.root/'key.txt'; self.key.write_text('SYNTHETIC-KEY')
        self.env = dict(SUPERAGENT_PLANNER_KEY_FILE=str(self.key), SUPERAGENT_PLANNER_LEDGER=str(self.ledger.path),
                        SUPERAGENT_TASK_BUDGET_ID='new_scope', SUPERAGENT_ACCEPTED_HISTORY_FILE=str(self.path),
                        SUPERAGENT_ACCEPTED_HISTORY_SHA256=sha(self.path), SUPERAGENT_CAMPUS_DIRECT_IP=CAMPUS_IP,
                        SUPERAGENT_OFFLINE='0')

    def write(self):
        self.path.write_text(json.dumps(self.record),encoding='utf-8')

    def account(self):
        return AcceptedHistoryAccount(self.ledger,'new_scope','run',self.path,sha(self.path))

    def test_explicit_configuration_builds_real_account_without_modifying_history(self):
        before=sha(self.ledger.path)
        with patch.dict(os.environ,self.env,clear=True):client=configured_flash(self.root)
        self.assertIsInstance(client.account,AcceptedHistoryAccount)
        self.assertEqual(client.campus_ip,CAMPUS_IP);self.assertEqual(client.max_requests,2)
        self.assertEqual(client.account.snapshot()['course_budget_compliance'],'unverified_history')
        self.assertEqual(sha(self.ledger.path),before)

    def test_default_strict_still_refuses_unresolved_project(self):
        env={k:v for k,v in self.env.items() if not k.startswith('SUPERAGENT_ACCEPTED_')}
        with patch.dict(os.environ,env,clear=True),self.assertRaisesRegex(ValueError,'unresolved'):
            configured_flash(self.root)

    def test_scope_hash_and_ledger_are_bound(self):
        for change in ({'allowed_scope':'other'}, {'ledger_path':str(self.root/'other.sqlite')}):
            with self.subTest(change=change):
                original=dict(self.record);self.record.update(change);self.write()
                with self.assertRaises(BudgetStateError):self.account().preflight()
                self.record=original;self.write()
        with self.assertRaisesRegex(BudgetStateError,'SHA'):
            AcceptedHistoryAccount(self.ledger,'new_scope','run',self.path,'0'*64)

    def test_partial_snapshot_configuration_is_rejected(self):
        env=dict(self.env);env.pop('SUPERAGENT_ACCEPTED_HISTORY_SHA256')
        with patch.dict(os.environ,env,clear=True),self.assertRaisesRegex(ValueError,'both'):
            configured_flash(self.root)

    def test_offline_remains_offline(self):
        with patch.dict(os.environ,{**self.env,'SUPERAGENT_OFFLINE':'1'},clear=True),self.assertRaisesRegex(ValueError,'offline'):
            configured_flash(self.root)

    def test_changed_or_removed_old_rows_block_before_reserving(self):
        account=self.account()
        with self.ledger._db() as db:
            db.execute("UPDATE requests SET payload='{}'");db.commit()
        with self.assertRaisesRegex(BudgetStateError,'modified'):account.reserve(.01,'deepseek-flash')
        with self.ledger._db() as db:
            db.execute('DELETE FROM requests');db.commit()
        with self.assertRaisesRegex(BudgetStateError,'changed'):account.reserve(.01,'deepseek-flash')

    def test_pending_and_new_unknown_stop_new_client(self):
        first=self.account();ticket=first.reserve(.01,'deepseek-flash')
        with self.assertRaisesRegex(BudgetStateError,'Pending'):self.account().preflight()
        first.unknown(ticket,'synthetic interruption')
        with self.assertRaisesRegex(BudgetStateError,'New or modified'):self.account().reserve(.01,'deepseek-flash')
        with self.ledger._db() as db:
            old=dict(db.execute('SELECT * FROM requests WHERE id=?',(self.old['id'],)).fetchone())
        self.assertEqual(old,self.old);self.assertIsNone(old['course_units'])

    def test_request_limit_is_cumulative_across_clients(self):
        for _ in range(2):
            account=self.account();ticket=account.reserve(.01,'deepseek-flash');account.settle(ticket,'deepseek-flash',10,5)
        with self.assertRaisesRegex(BudgetStateError,'request limit'):self.account().preflight()

    def test_cost_stop_threshold_is_cumulative_across_clients(self):
        account=self.account();ticket=account.reserve(.01,'deepseek-flash');account.settle(ticket,'deepseek-flash',240000,0)
        with self.assertRaisesRegex(BudgetStateError,'stop threshold'):self.account().reserve(.02,'deepseek-flash')

    def test_concurrent_reservations_admit_only_one(self):
        def attempt(_):
            try:return self.account().reserve(.01,'deepseek-flash')
            except BudgetStateError:return None
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(attempt,range(2)))
        self.assertEqual(sum(bool(v) for v in results),1)

    def test_course_limit_still_enforced(self):
        account=self.account();ticket=account.reserve(.01,'deepseek-flash');account.settle(ticket,'deepseek-flash',21000000,0)
        with self.assertRaisesRegex(BudgetStateError,'course limit'):account.reserve(.01,'deepseek-flash')

    def test_empty_new_project_is_not_an_accepted_history_substitute(self):
        empty=TaskBudgetLedger.initialize(self.root/'empty.sqlite',new_project=True)
        self.record.update(ledger_path=str(empty.path),accepted_unknown_requests=[]);self.write()
        with self.assertRaises(BudgetStateError):
            AcceptedHistoryAccount(empty,'new_scope','run',self.path,sha(self.path))

    def test_live_client_failure_is_one_request_then_persistently_blocked(self):
        with patch.dict(os.environ,self.env,clear=True):client=configured_flash(self.root)
        (self.root/'run').mkdir();logger=RunLogger(self.root/'run');logger.initialize_usage_ledger()
        with patch('superagent.experiment_llm.requests.Session') as factory:
            session=factory.return_value.__enter__.return_value
            session.post.side_effect=requests.Timeout('synthetic transport failure')
            with self.assertRaisesRegex(RuntimeError,'batch calls stopped'):client.ask({},logger,'task_semantic_interpretation')
            with self.assertRaisesRegex(RuntimeError,'circuit'):client.ask({},logger,'task_semantic_interpretation')
            self.assertEqual(session.post.call_count,1)
            self.assertEqual(session.post.call_args.kwargs['timeout'],(10,60))
            self.assertFalse(session.post.call_args.kwargs['allow_redirects'])
        with patch.dict(os.environ,self.env,clear=True),self.assertRaises(BudgetStateError):configured_flash(self.root)


class CampusTransportTests(unittest.TestCase):
    def test_connection_selects_only_socket_address_and_restores_hostname(self):
        connection=CampusConnection(HOST,443,timeout=7)
        with patch('urllib3.util.connection.create_connection',return_value=Mock()) as dial:
            connection._new_conn()
        self.assertEqual(dial.call_args.args[0],(CAMPUS_IP,443))
        self.assertEqual(connection.host,HOST);self.assertEqual(connection._dns_host,HOST)

    def test_tls_still_uses_original_hostname(self):
        connection=CampusConnection(HOST,443,timeout=7,cert_reqs='CERT_REQUIRED')
        with patch('urllib3.util.connection.create_connection',return_value=Mock()),patch('urllib3.connection._ssl_wrap_socket_and_match_hostname') as wrap:
            wrap.return_value.is_verified=True;connection.connect()
        self.assertEqual(wrap.call_args.kwargs['server_hostname'],HOST)
        self.assertEqual(wrap.call_args.kwargs['cert_reqs'],'CERT_REQUIRED')

    def test_other_origins_and_addresses_cannot_use_campus_transport(self):
        from superagent.campus_transport import validate_address
        for endpoint,address in [(ENDPOINT,'1.2.3.4'),('https://example.invalid',CAMPUS_IP)]:
            with self.assertRaises(ValueError):validate_address(endpoint,address)
        with self.assertRaises(ValueError):CampusConnection('example.invalid',443)._new_conn()

    def test_adapter_is_local_and_has_no_retries(self):
        from urllib3.poolmanager import pool_classes_by_scheme
        original=pool_classes_by_scheme['https'];adapter=CampusAdapter(max_retries=0)
        self.assertIs(pool_classes_by_scheme['https'],original)
        self.assertEqual(adapter.max_retries.total,0)
        pool=adapter.poolmanager.connection_from_url(ENDPOINT)
        self.assertIs(pool.ConnectionCls,CampusConnection)

    def test_dial_failure_restores_hostname_without_retry(self):
        connection=CampusConnection(HOST,443)
        from urllib3.exceptions import NewConnectionError
        with patch('urllib3.util.connection.create_connection',side_effect=OSError('synthetic')) as dial:
            with self.assertRaises(NewConnectionError):connection._new_conn()
        self.assertEqual(dial.call_count,1);self.assertEqual(connection._dns_host,HOST)


if __name__=='__main__':unittest.main()
