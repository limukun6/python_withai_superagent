from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from superagent.cli import build_parser,_run
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.pricing import resolve_pricing
from superagent.task_budget import TaskBudgetLedger,TaskBudgetAccount,BudgetStateError
from superagent.usage_audit import money_units


def history_row(key,scope,tokens=None):
    price=resolve_pricing('deepseek-flash');cost=price.course_cost(tokens,0) if tokens is not None else None
    return {'request_id':hashlib.sha256(key.encode()).hexdigest(),'scope':scope,'stage':'synthetic_fixture',
        'cost_status':'unknown' if tokens is None else 'known','course_cost_units':money_units(cost) if cost is not None else None,
        'course_cost_rmb':cost,'input_tokens':tokens,'output_tokens':0 if tokens is not None else None,
        'served_model':'deepseek-flash','requested_model':'deepseek-flash','pricing':price.metadata(),
        'actual_account_debit_rmb':None,'provider_response_id':'shared-id-is-not-a-dedup-key'}


class TaskBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.path=self.root/'budget.sqlite'

    def history(self):
        rows=[history_row('a','task1',1_000_000),history_row('b','task1'),history_row('c','task2',2_000_000),
              history_row('d','shared_diagnostics',1_000_000)]
        path=self.root/'audit.json';path.write_text(json.dumps({'calls':rows}));return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_import_groups_shared_cost_without_double_counting_project_spend(self):
        path,digest=self.history();ledger=TaskBudgetLedger.initialize(self.path,history=path,expected_sha256=digest)
        self.assertEqual(ledger.project_summary()['known_course_cost_rmb'],4.)
        self.assertEqual(ledger.project_summary()['unresolved_requests'],1)
        self.assertIsNone(ledger.project_summary()['complete_course_cost_rmb'])
        state=ledger.preflight('task2',.1)
        self.assertEqual(state['known_task_cost_rmb'],2.);self.assertEqual(state['shared_known_cost_rmb'],1.)
        self.assertEqual(state['remaining_course_allowance_rmb'],17.)
        with self.assertRaisesRegex(BudgetStateError,'unresolved'):ledger.preflight('task1',.1)
        ledger.import_history(path,digest)
        self.assertEqual(ledger.project_summary()['requests'],4)
        self.assertIsNone(state['actual_account_debit_rmb'])

    def test_conflicting_import_is_atomic_and_existing_ledger_cannot_be_reset(self):
        path,digest=self.history();ledger=TaskBudgetLedger.initialize(self.path,history=path,expected_sha256=digest)
        changed=json.loads(path.read_text());changed['calls'][0]=history_row('a','task1',1_000_001)
        path.write_text(json.dumps(changed));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(BudgetStateError,'Conflicting'):ledger.import_history(path,digest)
        self.assertEqual(ledger.project_summary()['known_course_cost_rmb'],4.)
        with self.assertRaises(FileExistsError):TaskBudgetLedger.initialize(self.path,new_project=True)
        self.assertEqual(ledger.project_summary()['known_course_cost_rmb'],4.)

    def test_pending_request_survives_new_process_object_and_settlement_restores_known_allowance(self):
        ledger=TaskBudgetLedger.initialize(self.path,new_project=True)
        ticket=ledger.reserve('task2',.01,'run-1','deepseek-flash')
        restarted=TaskBudgetLedger(self.path)
        with self.assertRaisesRegex(BudgetStateError,'pending'):restarted.preflight('task2',.1)
        state=restarted.settle(ticket,'deepseek-flash',100,20)
        self.assertEqual(state['pending_requests'],0);self.assertAlmostEqual(state['known_task_cost_rmb'],.00018)
        self.assertAlmostEqual(restarted.preflight('task2',.1)['remaining_course_allowance_rmb'],19.99982)
        with self.assertRaisesRegex(BudgetStateError,'overwritten'):restarted.settle(ticket,'deepseek-flash',100,20)

    def test_concurrent_reservations_cannot_both_pass_an_unsettled_scope(self):
        TaskBudgetLedger.initialize(self.path,new_project=True)
        def reserve(index):
            try:return TaskBudgetLedger(self.path).reserve('task2',.01,f'run-{index}','deepseek-flash')
            except BudgetStateError:return None
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(reserve,[1,2]))
        self.assertEqual(sum(r is not None for r in results),1)
        self.assertEqual(TaskBudgetLedger(self.path).snapshot('task2')['pending_requests'],1)

    def test_actual_overrun_is_recorded_not_clipped_and_later_calls_are_blocked(self):
        ledger=TaskBudgetLedger.initialize(self.path,new_project=True)
        ticket=ledger.reserve('task2',.01,'fixture','deepseek-flash')
        state=ledger.settle(ticket,'deepseek-flash',21_000_000,0)
        self.assertEqual(state['known_task_cost_rmb'],21.);self.assertTrue(state['limit_exceeded'])
        with self.assertRaisesRegex(BudgetStateError,'insufficient'):ledger.preflight('task2',.01)

    def test_unknown_import_and_bad_hash_are_not_silently_repaired(self):
        path,digest=self.history()
        with self.assertRaisesRegex(BudgetStateError,'SHA256'):
            TaskBudgetLedger.initialize(self.path,history=path,expected_sha256='bad')
        self.assertFalse(self.path.exists())
        value=json.loads(path.read_text());value['calls'][1]['course_cost_units']=0;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(BudgetStateError,'zero'):
            TaskBudgetLedger.initialize(self.path,history=path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(self.path.exists())

    def test_cli_blocks_old_unknown_before_data_credential_or_transport_access(self):
        path,digest=self.history();TaskBudgetLedger.initialize(self.path,history=path,expected_sha256=digest)
        args=build_parser().parse_args(['run','absent.zip','--planner','llm','--api-url','https://example.invalid',
            '--api-key-file','ABSENT-SECRET','--model','deepseek-flash','--budget-ledger',str(self.path),'--task-budget-id','task1'])
        with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),patch('superagent.cli.load_development_task') as data,patch('superagent.cli.ExperimentLLM') as client:
            with self.assertRaisesRegex(BudgetStateError,'unresolved'):_run(args)
            data.assert_not_called();client.assert_not_called()


class PersistentClientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.ledger=TaskBudgetLedger.initialize(self.root/'cost.sqlite',new_project=True)
        self.account=TaskBudgetAccount(self.ledger,'task2','synthetic-run')
        self.key=self.root/'dummy-key';self.key.write_text('SYNTHETIC-KEY')
        self.logger=RunLogger(self.root);self.logger.initialize_usage_ledger()

    def client(self):
        return ExperimentLLM('https://example.invalid',str(self.key),'deepseek-flash',.1,
            allowed_served_models={'deepseek-flash'},max_requests=2,account=self.account)

    def response(self,post,**changes):
        value={'model':'deepseek-flash','usage':{'prompt_tokens':100,'completion_tokens':20},
               'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
        value.update(changes);post.return_value.ok=True;post.return_value.json.return_value=value

    def test_request_is_reserved_before_transport_then_settled_and_linked_to_usage(self):
        client=self.client()
        with patch('superagent.experiment_llm.requests.post') as post:
            self.response(post)
            def inspect(*args,**kwargs):
                self.assertEqual(self.ledger.snapshot('task2')['pending_requests'],1)
                return post.return_value
            post.side_effect=inspect;client.ask({},self.logger,'planner')
        rows=[json.loads(line) for line in self.logger.usage_path.read_text().splitlines()]
        self.assertTrue(rows[-1]['budget_request_id']);self.assertTrue(rows[-1]['persistent_settlement_complete'])
        self.assertAlmostEqual(self.ledger.snapshot('task2')['known_task_cost_rmb'],.00018)

    def test_wrong_model_remains_unresolved_across_new_clients(self):
        with patch('superagent.experiment_llm.requests.post') as post:
            self.response(post,model='unapproved')
            with self.assertRaises(RuntimeError):self.client().ask({},self.logger,'planner')
            with self.assertRaises(BudgetStateError):self.client().ask({},self.logger,'planner')
            self.assertEqual(post.call_count,1)
        state=self.ledger.snapshot('task2');self.assertEqual(state['unresolved_requests'],1)
        self.assertIsNone(state['remaining_course_allowance_rmb'])

    def test_settlement_failure_keeps_known_usage_log_and_persistent_pending_block(self):
        client=self.client()
        with patch('superagent.experiment_llm.requests.post') as post,patch.object(self.account,'settle',side_effect=OSError('synthetic database failure')):
            self.response(post)
            with self.assertRaisesRegex(RuntimeError,'settlement failed'):client.ask({},self.logger,'planner')
        self.assertTrue(client.blocked)
        record=json.loads(self.logger.usage_path.read_text().splitlines()[-1])
        self.assertAlmostEqual(record['course_cost_rmb'],.00018);self.assertFalse(record['persistent_settlement_complete'])
        self.assertEqual(self.ledger.snapshot('task2')['pending_requests'],1)


if __name__=='__main__':unittest.main()
