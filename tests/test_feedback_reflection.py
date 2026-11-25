import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from superagent import feedback_reflection as review
from superagent.cli import build_parser
from superagent.task_budget import TaskBudgetLedger
from superagent.unified import reflection_client


def checkpoint():
    methods = {'existing': {'estimator': 'linear', 'representation': 'raw', 'classifier_fits': 1},
               'challenger': {'estimator': 'tree', 'representation': 'raw', 'classifier_fits': 1}}
    menu = [
        dict(action='confirm', members=['existing'], split='confirmation_1', utility=.6, evidence={}),
        dict(action='explore', members=['challenger'], split='primary', utility=.59,
             evidence=dict(novelty=.8, expected=.6, reserved_refits=1)),
    ]
    history = [dict(trial=i, candidate='existing', status='completed', split_id='primary',
                    search_score=.69 + .01*i) for i in (1,2,3)]
    profile = dict(statistics=dict(train_rows=900, class_count=2, raw_document='PRIVATE_SAMPLE'),
                   minority_class_fraction=.3)
    return review.evidence_packet(menu, methods, profile, history, 'macro_f1', 6,
                                  {'different':dict(improved=4, tied=0, worse=15, tasks=6)})


class FeedbackReflectionTests(unittest.TestCase):
    def test_local_can_change_one_action_and_predict_its_internal_outcome(self):
        p = checkpoint()
        choice = review.validate(review.local_review(p), p)
        self.assertEqual((choice['verdict'], choice['option'], choice['supporting_trials']), ('experiment', 1, [3]))
        self.assertAlmostEqual(choice['minimum_score'], .72)
        self.assertNotIn('PRIVATE_SAMPLE', json.dumps(p))
        p['remaining_fits'] = 1
        self.assertEqual(review.local_review(p)['option'], 0)
        p['earlier_primary_explorations']['different']['improved'] = 0
        self.assertEqual(review.local_review(p)['verdict'], 'insufficient')

    def test_first_eligible_checkpoint_needs_real_trials_and_an_alternative(self):
        rows = [dict(status='completed') for _ in range(3)]
        menu = [dict(action='explore'),dict(action='stop')]
        self.assertTrue(review.eligible(rows, menu))
        self.assertFalse(review.eligible(rows[:2]+[dict(status='failed')], menu))
        self.assertFalse(review.eligible(rows, menu[:1]))
        self.assertFalse(review.eligible(rows, [dict(action='confirm'),dict(action='stop')]))

    def test_invalid_or_unavailable_advice_keeps_incumbent_action(self):
        packet = checkpoint()
        for edits in [dict(supporting_trials=[4]),dict(option=-1),dict(minimum_score=float('nan')),
                      dict(refute_if=''),dict(option=1,verdict='keep'),dict(reason='')]:
            reply = {**review.local_review(packet), **edits}
            client = Mock()
            client.ask.return_value = reply
            result = review.decide('flash', packet, client, Mock())
            self.assertEqual((result['status'], result['decision']['option']), ('fallback', 0))
            self.assertEqual(client.ask.call_count, 1)
            self.assertEqual(client.ask.call_args.args[-1], 'feedback_reflection')
        self.assertEqual(review.decide('flash', packet, None, Mock())['status'], 'fallback')
        for mode in ('control','local'):
            client = Mock()
            self.assertEqual(review.decide(mode, packet, client, Mock())['status'], 'valid')
            client.ask.assert_not_called()

    def test_history_only_accepts_aggregate_evidence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'counts.json'
            counts = dict(improved=1, tied=0, worse=2, tasks=2)
            path.write_text(json.dumps(dict(near=counts, different=counts)))
            self.assertEqual(review.load_memory(path)['near'], counts)
            for edit in [dict(raw_rows=['x']),dict(tasks=4),dict(improved=True),dict(worse=-1)]:
                path.write_text(json.dumps(dict(near={**counts, **edit}, different=counts)))
                with self.assertRaises(ValueError):
                    review.load_memory(path)

    def test_cli_uses_dedicated_client_only_for_live_reflection(self):
        with patch('superagent.unified.configured_flash') as standard, patch('superagent.unified.reflection_client') as live, \
                patch('superagent.autonomous_run.train') as train:
            for mode in ('control','local','flash'):
                args = build_parser().parse_args(['unified','train.zip','--run-dir','run','--feedback-reflection',mode])
                args.handler(args)
                expected = live.return_value if mode=='flash' else None
                self.assertIs(train.call_args.kwargs['flash'], expected)
                self.assertEqual(train.call_args.kwargs['feedback_reflection'], mode)
            standard.assert_not_called()
            live.assert_called_once()

    def test_research_client_requires_marked_settled_history_and_unused_scope(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'outbox.sqlite'
            ledger = TaskBudgetLedger.initialize(path, new_project=True)
            key = Path(directory)/'dummy.txt'
            key.write_text('offline-test')
            env = dict(SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(path), SUPERAGENT_TASK_BUDGET_ID='fresh',
                       SUPERAGENT_PLANNER_KEY_FILE=str(key))
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    reflection_client(directory)
                with sqlite3.connect(path) as db:
                    db.executemany('INSERT INTO metadata VALUES (?,?)',
                        [('ledger_kind','provisional_local_outbox'),('team_merge_state','pending_team_handoff')])
                previous = ledger.reserve('past', .01, 'prior-test', 'DeepSeek-V4.1-Flash')
                with self.assertRaises(ValueError):
                    reflection_client(directory)
                ledger.settle(previous, 'DeepSeek-V4.1-Flash', 10, 10)
                client = reflection_client(directory)
                self.assertEqual(client.max_requests, 1)
                self.assertAlmostEqual(client.budget, .12)
                for extra in [dict(SUPERAGENT_TASK_BUDGET_ID='past'),dict(SUPERAGENT_OFFLINE='1')]:
                    with patch.dict(os.environ, extra), self.assertRaises(ValueError):
                        reflection_client(directory)
