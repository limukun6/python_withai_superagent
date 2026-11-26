"""Joint A/B behavior: semantic intake, opening analysis and inherited planner controls."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from superagent.cli import build_parser
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.semantic_contract import resolve_training_task
from superagent.task_opening import TaskOpening
from tests.test_proactive_probe import inspection, output_proposal
from tests.test_output_recipe import package
from tests.test_semantic_contract import Client
from tests.test_task_opening import training_task


class SharedClient(Client):
    spent=0.0
    def ask(self,payload,log,role,**kwargs):
        if self.blocked or self.request_count>=self.max_requests:
            raise AssertionError('Joint client attempted an extra call')
        return super().ask(payload,log,role)


class SemanticAutonomyMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def args(self,*flags):
        return build_parser().parse_args(['unified','task.zip','--run-dir',str(self.root/'run'),*flags])

    def test_official_command_passes_both_feature_sets_with_the_same_client(self):
        args=self.args('--task-interpretation','proactive','--task-analysis','flash',
                       '--protected-self-review','--prediction-input','query.zip')
        client=Mock()
        with patch('superagent.unified.reflection_client',return_value=client) as factory, \
                patch('superagent.autonomous_run.train',return_value=0) as train:
            self.assertEqual(args.handler(args),0)
        factory.assert_called_once_with(args.run_dir,'opening-flash')
        options=train.call_args.kwargs
        self.assertIs(options['flash'],client)
        self.assertEqual(options['interpretation'],'proactive')
        self.assertEqual(options['feedback_reflection'],'opening-flash')
        self.assertEqual(options['self_review'],'protected')
        self.assertEqual(options['prediction_package'],'query.zip')

    def test_local_review_can_coexist_with_explicit_paid_semantic_intake(self):
        args=self.args('--task-interpretation','proactive','--task-analysis','local')
        with patch('superagent.unified.configured_flash') as configured, \
                patch('superagent.autonomous_run.train',return_value=0) as train:
            args.handler(args)
        configured.assert_called_once_with(args.run_dir)
        self.assertIs(train.call_args.kwargs['flash'],configured.return_value)
        self.assertEqual(train.call_args.kwargs['feedback_reflection'],'opening-local')

    def test_semantic_tool_plan_and_opening_share_three_requests(self):
        path=package(self.root/'task.zip')
        reply=dict(goal='Classify observations.',metric='accuracy',citations=['contract','shape'],
                   uncertainty='Evidence is limited.',check='keep',hypothesis='Keep the measured route.',
                   if_false='Retain the original delivery.')
        client=SharedClient([inspection(),output_proposal(),reply],limit=3)
        resolve_training_task(path,client,mode='proactive')
        self.assertEqual(client.request_count,2)
        task,folds=training_task()
        opening=TaskOpening('opening-flash',task,MetricSpec('accuracy',(0,1)),folds,[],'ordinary',
                            RunLogger(self.root),client,time.monotonic(),60)
        opening.launch(); opening.finish(); opening.launch(); opening.collect(0)
        self.assertEqual(client.request_count,3)
        self.assertEqual(opening.record['interpretation_status'],'valid')
        self.assertEqual(opening.record['prior_shared_requests'],2)

    def test_exhausted_semantic_allowance_cannot_launch_opening(self):
        task,folds=training_task(); client=SharedClient([],limit=3); client.request_count=3
        opening=TaskOpening('opening-flash',task,MetricSpec('accuracy',(0,1)),folds,[],'ordinary',
                            RunLogger(self.root),client,time.monotonic(),60)
        opening.launch(); opening.finish()
        self.assertEqual(client.request_count,3)
        self.assertEqual(opening.record['interpretation_status'],'unavailable')
        self.assertIn('Shared',opening.record['skip_reason'])

    def test_combined_official_intake_preserves_budget_and_contract_before_fit(self):
        from superagent.autonomous_run import _train
        client=SharedClient([output_proposal()]); path=package(self.root/'task.zip')
        class AtCheckpoint(RuntimeError): pass
        with patch('superagent.autonomous_run.standby',side_effect=AtCheckpoint) as checkpoint:
            with self.assertRaises(AtCheckpoint):
                _train(path,self.root/'combined',flash=client,interpretation='proactive',
                       feedback_reflection='opening-flash',self_review='protected',prediction_package=path,fit_limit=2)
        task=checkpoint.call_args.args[0]
        self.assertEqual(task.spec['output_recipe']['columns'][0]['name'],'Ticket')
        self.assertEqual(client.request_count,1)
        self.assertEqual(client.max_requests,3)
        record=json.loads((self.root/'combined/run_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(record['task_interpretation'],'proactive')
        self.assertEqual(record['feedback_reflection'],'opening-flash')
        self.assertEqual(record['self_review'],'protected')

    def test_b_original_opening_single_call_limit_and_explicit_zero_remain(self):
        from superagent.autonomous_run import _train
        for initial,expected in [(3,1),(0,0)]:
            client=SharedClient([],limit=initial)
            with (patch('superagent.semantic_contract.resolve_training_task',side_effect=ValueError('probe')),
                  patch('superagent.partial_classification.record_input_failure',return_value=1)):
                _train('input.zip',self.root/str(initial),flash=client,feedback_reflection='opening-flash')
            self.assertEqual(client.max_requests,expected)
            self.assertEqual(client.request_count,0)

    def test_solve_child_gets_a_and_b_flags_and_frozen_history(self):
        from unified_agent import solve_training
        from superagent.exploration_rule import snapshot
        rule=self.root/'rule.json'; rule.write_text(json.dumps(snapshot([0.,0.,0.])),encoding='utf-8')
        child=Mock(); child.wait.return_value=0
        run=self.root/'solve'
        with patch('subprocess.Popen',return_value=child) as popen:
            code=solve_training(self.root/'task.zip',run,fit_limit=10,time_limit=60,self_review='protected',
                                exploration_rule=rule,feedback_reflection='opening-local',
                                task_interpretation='proactive',prediction_input=self.root/'query.zip')
        self.assertEqual(code,0)
        flags=popen.call_args.args[0]
        for option,value in [('--task-interpretation','proactive'),('--feedback-reflection','opening-local'),
                             ('--prediction-input',str((self.root/'query.zip').resolve()))]:
            self.assertEqual(flags[flags.index(option)+1],value)
        self.assertIn('--protected-self-review',flags)
        self.assertEqual(Path(flags[flags.index('--exploration-rule')+1]),run/'exploration_rule.json')
        record=json.loads((run/'training_protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(record['task_interpretation'],'proactive')
        self.assertEqual(record['feedback_reflection'],'opening-local')


if __name__=='__main__': unittest.main()
