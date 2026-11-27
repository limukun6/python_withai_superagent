"""CLI auto review separates decodable inputs from complete task/output meaning."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np

from superagent.cli import build_parser
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.semantic_contract import resolve_training_task
from tests.test_semantic_contract import Client


def standard_case(path,manifest=False):
    output=dict(version='output-csv-v1',delimiter=';',encoding='utf-8-sig',header=True,
                columns=[dict(name='Ticket',kind='input_id',field='record_id'),
                         dict(name='Decision',kind='label',mapping={'accept':'Y','decline':'N'})],order=dict(mode='input'))
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('README.md','Classify label from signal. The primary metric is accuracy. Learn from train.csv and predict test.csv. '
            'Write semicolon-separated UTF-8 CSV with BOM and header Ticket;Decision. Ticket copies record_id, including leading zeros. '
            'Decision is Y for accept and N for decline.')
        archive.writestr('train.csv','signal,label\n'+''.join(f'{i},{"accept" if i%2 else "decline"}\n' for i in range(12)))
        archive.writestr('test.csv','record_id,signal\n0009,9\n0002,2\n')
        if manifest:archive.writestr('manifest.json',json.dumps(dict(schema_version=1,modality='tabular',target='label',metric='accuracy',
            columns=[dict(name='signal',type='numeric')],splits=dict(train=dict(format='csv',path='train.csv'),test=dict(format='csv',path='test.csv')))))
    refs=['doc:README.md:0']
    if manifest:answer=dict(version='semantic-output-v1',output=output,evidence_refs=refs,uncertainties=[])
    else:answer=dict(version='semantic-task-v2',plan=dict(training=dict(member='train.csv',collection=None,delimiter=','),
        prediction=dict(member='test.csv',collection=None,delimiter=','),target='label',columns=[dict(name='signal',type='numeric')],
        metric='accuracy',positive_label=None,split=None,output=output),evidence={k:refs for k in
            ['training','prediction','target','columns','metric','output']},uncertainties=[])
    return path,answer,output


class AutoSemanticsTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def test_local_decoding_success_is_not_complete_output_under_explicit_fallback(self):
        for manifest in [False,True]:
            path,answer,expected=standard_case(self.root/f'{manifest}.zip',manifest)
            client=Client([answer]);task=resolve_training_task(path,client,mode='fallback',prediction_package=path)
            self.assertEqual(client.request_count,0);self.assertNotIn('output_recipe',task.spec)
            self.assertEqual(len(task.y_train),12)

    def test_auto_reads_standard_table_output_with_one_model_reply(self):
        path,answer,expected=standard_case(self.root/'plain.zip');client=Client([answer])
        task=resolve_training_task(path,client,mode='auto',prediction_package=path)
        self.assertEqual(task.spec['output_recipe'],expected);self.assertEqual(client.request_count,1)

    def test_auto_manifest_uses_only_overlay_and_keeps_original_inputs(self):
        path,answer,expected=standard_case(self.root/'manifest.zip',True)
        original=load_training_task(path,read_declarations=True);client=Client([answer])
        task=resolve_training_task(path,client,mode='auto',prediction_package=path)
        np.testing.assert_array_equal(task.X_train,original.X_train);np.testing.assert_array_equal(task.y_train,original.y_train)
        self.assertEqual(task.spec['input_schema'],original.spec['input_schema'])
        self.assertEqual(task.spec['output_recipe'],expected)

    def test_manifest_input_change_is_not_permitted_by_auto(self):
        path,_,_=standard_case(self.root/'manifest.zip',True)
        _,answer,_=standard_case(self.root/'other.zip');answer['plan']['target']='invented'
        client=Client([answer],limit=1)
        with self.assertRaises(ValueError):resolve_training_task(path,client,mode='auto',prediction_package=path)
        self.assertEqual(client.request_count,1)

    def test_unconfigured_auto_uses_local_path_and_logs_that_review_was_not_run(self):
        path,_,_=standard_case(self.root/'local.zip');run=self.root/'run';run.mkdir()
        task=resolve_training_task(path,None,RunLogger(run),mode='auto',prediction_package=path)
        self.assertEqual(len(task.y_train),12);self.assertNotIn('output_recipe',task.spec)
        events=[json.loads(s) for s in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        selected=next(e for e in events if e['kind']=='semantic_mode_selected')
        self.assertEqual(selected['selected_mode'],'fallback');self.assertEqual(selected['reason'],'no_configured_model')

    def test_no_documents_does_not_invent_a_semantic_call(self):
        path,answer,_=standard_case(self.root/'nodoc.zip')
        with zipfile.ZipFile(path) as archive:members={n:archive.read(n) for n in archive.namelist() if n!='README.md'}
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
        client=Client([answer]);task=resolve_training_task(path,client,mode='auto',prediction_package=path)
        self.assertEqual(len(task.y_train),12);self.assertEqual(client.request_count,0)

    def test_blocked_or_exhausted_model_does_not_silently_certify_local_output(self):
        path,answer,_=standard_case(self.root/'task.zip')
        blocked=Client([answer]);blocked.blocked=True;empty=Client([answer],limit=0)
        for client in [blocked,empty]:
            with self.subTest(client=client),self.assertRaises(ValueError):resolve_training_task(path,client,mode='auto',prediction_package=path)
            self.assertEqual(client.request_count,0)

    def test_both_human_CLI_parsers_default_to_auto_and_allow_explicit_fallback(self):
        args=build_parser().parse_args(['unified','task.zip','--run-dir','unused'])
        self.assertEqual(args.task_interpretation,'auto')
        explicit=build_parser().parse_args(['unified','task.zip','--run-dir','unused','--task-interpretation','fallback'])
        self.assertEqual(explicit.task_interpretation,'fallback')
        import unified_agent
        with patch('sys.argv',['unified_agent','train','task.zip','--run-dir','unused']),patch('unified_agent.train',return_value=0) as train:
            self.assertEqual(unified_agent.main(),0)
        self.assertEqual(train.call_args.kwargs['task_interpretation'],'auto')

    def test_programmatic_defaults_stay_fallback_for_prior_experiments(self):
        import unified_agent
        self.assertEqual(inspect.signature(unified_agent.train).parameters['task_interpretation'].default,'fallback')
        self.assertEqual(inspect.signature(resolve_training_task).parameters['mode'].default,'fallback')

    def test_cached_auto_avoids_implicit_client_setup(self):
        args=build_parser().parse_args(['unified','task.zip','--run-dir','unused','--semantic-artifact-from','saved.json'])
        with patch('superagent.unified.configured_flash',side_effect=AssertionError('No implicit billing')),patch('superagent.autonomous_run.train',return_value=0) as train:
            self.assertEqual(args.handler(args),0)
        self.assertIsNone(train.call_args.kwargs['flash']);self.assertEqual(train.call_args.kwargs['interpretation'],'auto')

    def test_legacy_policy_keeps_its_old_reader_when_human_default_is_auto(self):
        import unified_agent
        with patch('unified_agent.train_legacy',return_value=0) as legacy:
            self.assertEqual(unified_agent.train('task.zip',self.root/'legacy',policy='legacy',task_interpretation='auto'),0)
        legacy.assert_called_once()
        path, _, _ = standard_case(self.root/'unrestricted.zip')
        args=build_parser().parse_args(['unified',str(path),'--run-dir','unused','--policy','legacy'])
        with patch('superagent.unified.legacy_command',return_value=0) as command:
            self.assertEqual(args.handler(args),0)
        command.assert_called_once()

    def test_auto_client_preflight_failure_is_not_marked_complete(self):
        from superagent.autonomous_run import _train
        path,_,_=standard_case(self.root/'task.zip')
        with patch('superagent.partial_classification.record_input_failure',return_value=2) as failure:
            code=_train(path,self.root/'failed',fit_limit=2,interpretation='auto',flash_setup_issue='synthetic ledger configuration unavailable')
        self.assertEqual(code,2);self.assertIsNone(failure.call_args.kwargs['resolved_task'])

    def test_auto_opening_retains_original_three_call_shared_ceiling(self):
        from superagent.autonomous_run import _train
        class StopBeforeFit(RuntimeError):pass
        path,answer,_=standard_case(self.root/'task.zip');client=Client([answer],limit=3)
        with patch('superagent.autonomous_run.standby',side_effect=StopBeforeFit):
            with self.assertRaises(StopBeforeFit):_train(path,self.root/'prefit',fit_limit=2,flash=client,
                interpretation='auto',feedback_reflection='opening-local',prediction_package=path)
        self.assertEqual(client.max_requests,3);self.assertEqual(client.request_count,1)


if __name__=='__main__':unittest.main()
