import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from superagent.benchmark import run_one
from superagent.cli import CAMPUS_CHAT_ENDPOINT, CAMPUS_FLASH_MODEL
from superagent.data import DevelopmentTask
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.task_budget import TaskBudgetAccount, TaskBudgetLedger


def sse_frames(candidate, include_usage=True, finish_reason='stop'):
    answer = json.dumps({'candidate': candidate, 'reason': 'Test an alternative regularization strength'})
    yield b'data: ' + json.dumps({'model': CAMPUS_FLASH_MODEL, 'choices': [
        {'index': 0, 'delta': {'content': answer}, 'finish_reason': finish_reason},
    ]}).encode()
    if include_usage:
        yield b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}'
    yield b'data: [DONE]'


class ExpandedFeedbackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        key = self.root / 'dummy-key'
        key.write_text('TEST-KEY', encoding='utf-8')
        self.ledger = TaskBudgetLedger.initialize(self.root / 'budget.sqlite', new_project=True)
        account = TaskBudgetAccount(self.ledger, 'task1', 'fixture')
        self.client = ExperimentLLM(CAMPUS_CHAT_ENDPOINT, key, CAMPUS_FLASH_MODEL, .1,
                                    allowed_served_models={CAMPUS_FLASH_MODEL}, account=account,
                                    max_requests=3, streaming=True)

    @patch('superagent.experiment_llm.requests.Session')
    def test_scalar_feedback_runs_shared_anchor_and_next_candidate(self, session):
        connection = session.return_value.__enter__.return_value
        response = connection.post.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_lines.side_effect = [sse_frames('logistic_C0.25'), sse_frames('logistic_C1.0')]
        x = np.array([[str(i), str(i % 4)] for i in range(80)])
        task = DevelopmentTask('categorical_tabular', 'fixture', x, np.arange(80) % 2, None, None, {})
        folder = self.root / 'trace'
        summary = run_one(task, 'scalar_feedback', 707, folder, self.client, max_trials=3,
                          anchor_first=True, portfolio='expanded')
        self.assertTrue(summary['valid_policy_run'])
        self.assertEqual(summary['trials_attempted'], 3)
        self.assertEqual(summary['llm_calls'], 2)
        self.assertAlmostEqual(summary['course_cost_rmb'], .00036)
        self.assertFalse(connection.trust_env)
        first, second = [call.kwargs for call in connection.post.call_args_list]
        self.assertTrue(first['stream'])
        self.assertFalse(first['allow_redirects'])
        self.assertLessEqual(first['timeout'][1], 60)
        for request, count in ((first, 1), (second, 2)):
            prompt = json.loads(request['json']['messages'][-1]['content'])
            self.assertEqual(prompt['remaining_trials'], 3 - count)
            self.assertEqual(len(prompt['observed_trials']), count)
            for row in prompt['observed_trials']:
                self.assertEqual(set(row), {'candidate', 'status', 'search_score'})
                self.assertIsInstance(row['search_score'], float)
        outcomes = json.loads((folder / 'trials.json').read_text(encoding='utf-8'))
        self.assertEqual([row['candidate'] for row in outcomes], ['logistic_C0.05', 'logistic_C0.25', 'logistic_C1.0'])

    @patch('superagent.experiment_llm.requests.Session')
    def test_missing_stream_usage_blocks_further_calls(self, session):
        connection = session.return_value.__enter__.return_value
        response = connection.post.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_lines.return_value = sse_frames('logistic_C0.25', include_usage=False)
        logger = RunLogger(self.root)
        logger.initialize_usage_ledger()
        with self.assertRaisesRegex(RuntimeError, 'lacks valid token usage'):
            self.client.ask({}, logger, 'scalar_feedback')
        with self.assertRaisesRegex(RuntimeError, 'circuit open'):
            self.client.ask({}, logger, 'scalar_feedback')
        self.assertEqual(connection.post.call_count, 1)
        self.assertFalse(self.ledger.snapshot('task1')['accounting_complete'])

    @patch('superagent.experiment_llm.requests.Session')
    def test_truncated_stream_keeps_usage_and_never_returns_a_decision(self, session):
        connection = session.return_value.__enter__.return_value
        response = connection.post.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_lines.return_value = sse_frames('logistic_C0.25', finish_reason='length')
        logger = RunLogger(self.root)
        logger.initialize_usage_ledger()
        with self.assertRaisesRegex(ValueError, 'Output token limit reached'):
            self.client.ask({}, logger, 'scalar_feedback')
        self.assertEqual(connection.post.call_count, 1)
        state = self.ledger.snapshot('task1')
        self.assertTrue(state['accounting_complete'])
        self.assertAlmostEqual(state['known_task_cost_rmb'], .00018)
