"""Actual CLI/client/accounting integration with an offline HTTP transport."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock

import requests
from superagent.cli import build_parser,_run,_validate,CAMPUS_FLASH_MODEL
from superagent.task_budget import TaskBudgetLedger,BudgetStateError
from superagent.pricing import resolve_pricing
from tests.test_unified_cli import fixture


class CheckedCLITests(unittest.TestCase):
    def setUp(self):
        parent=os.environ.get('A1_04_ARTIFACT_DIR')
        if parent:
            self.root=Path(parent)/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        else:
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.path=self.root/'fixture.zip';fixture(self.path)
        self.key=self.root/'fixture_credential.txt';self.key.write_text('OFFLINE-SYNTHETIC-CREDENTIAL',encoding='utf-8')
        self.ledger=TaskBudgetLedger.initialize(self.root/'ledger.sqlite',new_project=True)
        self.calls=0

    def tearDown(self):
        fits=0
        for run in self.root.glob('run*'):
            outcome=run/'run_outcome.json'
            if outcome.is_file():
                value=json.loads(outcome.read_text(encoding='utf-8'))
                fits+=value.get('classifier_fit_accounting',{}).get('completed_classifier_fits',value.get('completed_search_fits',0))
        (self.root/'stats.json').write_text(json.dumps(dict(mock_http_requests=self.calls,classifier_fits=fits)),encoding='utf-8')

    def args(self,planner,name,extras=()):
        return build_parser().parse_args(['run',str(self.path),'--planner',planner,'--api-key-file',str(self.key),
            '--budget-ledger',str(self.ledger.path),'--task-budget-id','fixture','--run-dir',str(self.root/name),
            '--trials','2','--time-limit','180',*extras])

    def transport(self,mode='ok'):
        def post(url,**kwargs):
            self.calls+=1
            if mode=='timeout':raise requests.Timeout('controlled timeout')
            payload=json.loads(kwargs['json']['messages'][1]['content'])
            candidate=payload['available_candidates'][0]
            answer=dict(candidate=candidate['id'],task_kind=payload['task_spec']['kind'],
                configuration_claim={'classifier':payload['candidate_facts'][candidate['id']]['classifier']},
                reason='Controlled experiment.',review='Read the previous result.',hypothesis='Try this alternative.',
                evidence_trials=[1],expected_change=dict(metric='search_score',reference_trial=1,direction='increase',min_delta=.001),slice_expectation=None)
            content=json.dumps(answer) if mode!='bad_json' else 'invalid JSON'
            chunk=dict(id='offline-fixture-response',model=CAMPUS_FLASH_MODEL,
                choices=[dict(index=0,delta={'content':content},finish_reason='stop')])
            if mode!='missing_usage':chunk['usage']={'prompt_tokens':100,'completion_tokens':20}
            response=MagicMock();response.status_code=200;response.ok=True
            response.iter_lines.return_value=[b'data: '+json.dumps(chunk).encode(),b'data: [DONE]']
            response.__enter__.return_value=response
            self.assertEqual(kwargs['timeout'],(10,60))
            return response
        session=MagicMock();session.__enter__.return_value=session;session.post.side_effect=post
        return patch('superagent.experiment_llm.requests.Session',return_value=session)

    def test_both_modes_finish_and_accumulate_same_task_ledger(self):
        with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),self.transport(),contextlib.redirect_stdout(io.StringIO()):
            for planner,name in [('raw-history','run_raw'),('checked-history','run_checked')]:
                self.assertEqual(_run(self.args(planner,name)),0)
                outcome=json.loads((self.root/name/'run_outcome.json').read_text(encoding='utf-8'))
                self.assertTrue(outcome['requested_planner_completed']);self.assertEqual(outcome['new_api_requests'],1)
                self.assertEqual(outcome['classifier_fit_accounting']['completed_classifier_fits'],7)
                self.assertTrue((self.root/name/'selected_model.joblib').is_file())
                manifest=json.loads((self.root/name/'run_manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['maximum_api_requests'],1)
                self.assertEqual(manifest['planner_failure_policy'],'stop_without_retry_or_fallback')
                events=[json.loads(s) for s in (self.root/name/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                request=next(r for r in events if r['kind']=='llm_request')
                context=json.loads(request['messages'][1]['content'])
                self.assertEqual('checks' in context,planner=='checked-history')
                self.assertTrue(request['streaming'])
        snapshot=self.ledger.snapshot('fixture')
        self.assertAlmostEqual(snapshot['known_task_cost_rmb'],2*resolve_pricing(CAMPUS_FLASH_MODEL).course_cost(100,20))
        self.assertTrue(snapshot['accounting_complete']);self.assertEqual(self.calls,2)

    def failure(self,mode,unknown):
        with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),self.transport(mode):
            with self.assertRaises((RuntimeError,ValueError)):_run(self.args('checked-history','run_failure'))
        self.assertEqual(self.calls,1)
        value=json.loads((self.root/'run_failure/run_outcome.json').read_text(encoding='utf-8'))
        self.assertFalse(value['requested_planner_completed']);self.assertEqual(value['new_api_requests'],1)
        self.assertTrue((self.root/'run_failure/trial_1.joblib').is_file())
        events=[json.loads(s) for s in (self.root/'run_failure/events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertFalse(any(r['kind']=='planner_fallback' for r in events))
        self.assertEqual(self.ledger.snapshot('fixture')['accounting_complete'],not unknown)
        if unknown:
            self.assertIsNone(value['complete_course_cost_rmb'])
            with self.assertRaises(BudgetStateError):self.ledger.preflight('fixture',.1)
        else:self.assertGreater(value['known_course_cost_rmb'],0)

    def test_transport_error_stops_and_blocks_task_account(self):self.failure('timeout',True)
    def test_missing_usage_stops_and_keeps_cost_unknown(self):self.failure('missing_usage',True)
    def test_invalid_json_keeps_known_usage_but_stops(self):self.failure('bad_json',False)

    def test_incompatible_features_rejected_before_loading_task(self):
        for extras in (['--architecture','self_review'],['--refinement','research'],['--allocation','successive']):
            with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}),patch('superagent.cli.load_development_task') as load:
                with self.assertRaises(ValueError):_run(self.args('checked-history','run_unused',extras))
                load.assert_not_called()

if __name__=='__main__':unittest.main()
