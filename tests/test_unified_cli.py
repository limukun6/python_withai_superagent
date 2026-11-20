import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.cli import build_parser,_run,_validate
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM


def fixture(path):
    spec={'schema_version':1,'modality':'tabular','metric':'accuracy',
          'splits':{s:{'path':s+'.npz','format':'npz'} for s in ('train','validation')}}
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('manifest.json',json.dumps(spec))
        for name,n in [('train',60),('validation',15)]:
            buf=io.BytesIO();np.savez_compressed(buf,X=np.arange(n*3).reshape(n,3),y=np.arange(n)%3)
            archive.writestr(name+'.npz',buf.getvalue())


class UnifiedCLITests(unittest.TestCase):
    def test_both_names_have_identical_handler_and_local_defaults(self):
        parser=build_parser()
        a=parser.parse_args(['run','fixture.zip']);b=parser.parse_args(['experiment','fixture.zip'])
        self.assertIs(a.handler,b.handler)
        self.assertEqual({k:v for k,v in vars(a).items() if k!='command'},
                         {k:v for k,v in vars(b).items() if k!='command'})
        self.assertEqual(a.planner,'local');self.assertIsNone(_validate(a))

    def test_invalid_api_options_fail_before_credential_or_data_access(self):
        cases=[['--api-key-file','secret'],['--trials','0'],['--time-limit','nan'],
               ['--trial-timeout','inf'],['--planner','llm'],
               ['--planner','llm','--api-url','http://example.invalid','--api-key-file','secret','--model','deepseek-flash'],
               ['--planner','llm','--api-url','https://example.invalid','--api-key-file','secret','--model','unapproved'],
               ['--planner','llm','--api-url','https://example.invalid','--api-key-file','secret','--model','deepseek-flash','--llm-budget-rmb','nan'],
               ['--planner','llm','--api-url','https://example.invalid','--api-key-file','secret','--model','deepseek-flash','--llm-budget-rmb','21']]
        with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}):
            for flags in cases:
                with self.subTest(flags=flags),patch('superagent.cli.load_development_task') as data,patch('superagent.cli.ExperimentLLM') as client:
                    args=build_parser().parse_args(['run','missing.zip',*flags])
                    with self.assertRaises(ValueError):_run(args)
                    data.assert_not_called();client.assert_not_called()

    def test_local_never_constructs_client_and_partial_outputs_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'fixture.zip';fixture(path)
            def fake(task,strategy,seed,directory,client,**kwargs):
                self.assertIsNone(client)
                self.assertTrue(strategy.startswith('local_'))
                self.assertEqual(kwargs['run_metadata']['selection_data'],'training-derived partitions only')
                directory.mkdir();RunLogger(directory).initialize_usage_ledger()
                return {'selected_candidate':'dummy','valid_policy_run':True,'usage_complete':True,
                    'trials_attempted':3,'confirmation':{'valid_confirmation':False},'refit_succeeded':False,
                    'official_validation_accuracy':.4,'fallback_count':0,'course_cost_rmb':0.}
            args=build_parser().parse_args(['run',str(path),'--run-dir',str(root/'out'),'--trials','3'])
            with patch('superagent.cli.ExperimentLLM') as client,patch('superagent.cli.run_one',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_run(args),2);client.assert_not_called()
            outcome=json.loads((root/'out/run_outcome.json').read_text(encoding='utf-8'))
            self.assertTrue(outcome['model_available']);self.assertFalse(outcome['requested_planner_completed'])
            self.assertEqual(outcome['new_api_requests'],0);self.assertEqual(outcome['complete_course_cost_rmb'],0)
            with patch('superagent.cli.ExperimentLLM') as client:
                with self.assertRaises(FileExistsError):_run(args)
                client.assert_not_called()


class APIBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.key=self.root/'dummy_key.txt';self.key.write_text('UNIT-TEST-KEY-NOT-REAL')
        self.logger=RunLogger(self.root);self.logger.initialize_usage_ledger()

    def client(self,**kwargs):
        return ExperimentLLM('https://example.invalid/api',str(self.key),'deepseek-flash',.1,
            max_requests=1,allowed_served_models={'DeepSeek-V4.1-Flash-1017'},**kwargs)

    def records(self):return [json.loads(s) for s in self.logger.usage_path.read_text().splitlines()][1:]

    def response(self,post,**changes):
        body={'model':'DeepSeek-V4.1-Flash-1017','usage':{'prompt_tokens':100,'completion_tokens':20},
              'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
        body.update(changes);post.return_value.ok=True;post.return_value.json.return_value=body

    def test_hard_request_limit_and_no_automatic_retry(self):
        client=self.client()
        with patch('superagent.experiment_llm.requests.post') as post:
            self.response(post);self.assertEqual(client.ask({},self.logger,'planner'),{})
            with self.assertRaisesRegex(RuntimeError,'request-count'):client.ask({},self.logger,'planner')
            self.assertEqual(post.call_count,1)
            self.assertFalse(post.call_args.kwargs['allow_redirects'])
        self.assertEqual(client.request_count,1);self.assertAlmostEqual(client.spent,.00018)

    def test_unexpected_model_keeps_tokens_but_no_price_and_opens_circuit(self):
        client=self.client()
        with patch('superagent.experiment_llm.requests.post') as post:
            self.response(post,model='unknown-model')
            with self.assertRaisesRegex(RuntimeError,'Unexpected served model'):client.ask({},self.logger,'planner')
            with self.assertRaisesRegex(RuntimeError,'circuit'):client.ask({},self.logger,'planner')
            self.assertEqual(post.call_count,1)
        record=self.records()[0]
        self.assertEqual(record['input_tokens'],100);self.assertEqual(record['output_tokens'],20)
        self.assertTrue(record['token_usage_known']);self.assertFalse(record['usage_known'])
        self.assertIsNone(record['course_cost_rmb']);self.assertIsNone(record['actual_cost_rmb'])
        self.assertTrue(client.blocked)

    def test_malformed_body_records_unknown_instead_of_disappearing(self):
        for bad in ([],None,{'usage':['bad']}):
            with self.subTest(body=bad),patch('superagent.experiment_llm.requests.post') as post:
                client=self.client();post.return_value.ok=True;post.return_value.json.return_value=bad
                with self.assertRaises(RuntimeError):client.ask({},self.logger,'planner')
                self.assertTrue(client.blocked)
                self.assertFalse(self.records()[-1]['usage_known'])
                self.assertIsNone(self.records()[-1]['course_cost_rmb'])

    def test_valid_usage_is_billed_even_when_response_has_no_decision(self):
        client=self.client()
        with patch('superagent.experiment_llm.requests.post') as post:
            self.response(post,choices=[])
            with self.assertRaisesRegex(ValueError,'Missing text'):client.ask({},self.logger,'planner')
        self.assertAlmostEqual(self.records()[0]['course_cost_rmb'],.00018)
        self.assertEqual(self.records()[0]['event'],'llm_call')

    def test_bad_constructor_parameters_do_not_read_key(self):
        for kwargs in ({'max_requests':0},{'max_requests':True},{'allowed_served_models':set()},
                       {'reasoning_effort':'unavailable'}):
            with self.subTest(kwargs=kwargs),patch.object(Path,'read_text') as read:
                with self.assertRaises(ValueError):
                    ExperimentLLM('https://example.invalid',str(self.key),'deepseek-flash',.1,**kwargs)
                read.assert_not_called()


if __name__=='__main__':unittest.main()
