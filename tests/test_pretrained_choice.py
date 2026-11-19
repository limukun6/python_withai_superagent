import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from superagent.evidence_agent import route_of
from superagent.pretrained_choice import CANDIDATE, MODEL, budget, consider
from superagent.task_budget import BudgetStateError, TaskBudgetLedger


class PretrainedTests(unittest.TestCase):
    def test_feedback_and_modality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = [{'trial':i,'candidate':f'word_C{i}','status':'completed','search_score':.5} for i in (1,2)]
            labels = np.zeros(32)
            for i in (1,2):
                joblib.dump({'search_predictions':np.ones(32)},root/f'trial_{i}.joblib')
            with patch('superagent.pretrained_choice.budget') as check:
                self.assertEqual(consider('numeric_tabular',history,root,labels,{})['decision'],'skip')
                check.assert_not_called()
                self.assertEqual(consider('pair_text',history,root,labels,{})['decision'],'try')
                check.assert_called_once()
                joblib.dump({'search_predictions':labels},root/'trial_2.joblib')
                self.assertEqual(consider('pair_text',history,root,labels,{})['decision'],'skip')
                self.assertEqual(consider('single_text',history[:1],root,labels,{})['decision'],'skip')
                self.assertEqual(route_of(CANDIDATE),'pretrained_qwen3_context')

    def test_real_token_accounting_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ledger.sqlite'
            ledger = TaskBudgetLedger.initialize(path,new_project=True)
            config = dict(ledger=path,scope='fixture',run='one',cap_rmb=.00003)
            key=budget(config,90,reserve=True)
            with self.assertRaises(BudgetStateError):
                budget(config,1)
            ledger.settle(key,MODEL,90,0)
            self.assertAlmostEqual(ledger.snapshot('fixture')['known_task_cost_rmb'],.000027)
            with self.assertRaises(BudgetStateError):
                budget(config,11,reserve=True)
            key=budget(config,10,reserve=True)
            ledger.mark_unknown(key,'interruption')
            with self.assertRaises(BudgetStateError):
                budget(config,1)

    def test_campaign_cap_survives_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ledger.sqlite'
            ledger = TaskBudgetLedger.initialize(path,new_project=True)
            first = dict(ledger=path,scope='fixture',run='pilot',cap_rmb=.1)
            ledger.settle(budget(first,100,reserve=True),MODEL,100,0)
            second = dict(first,run='replication',previous_run_ids=['pilot'],campaign_cap_rmb=.000033)
            with self.assertRaises(BudgetStateError):
                budget(second,11)
            ledger.settle(budget(second,10,reserve=True),MODEL,10,0)
            self.assertAlmostEqual(ledger.snapshot('fixture')['known_task_cost_rmb'],.000033)

    def test_a_single_new_candidate_cannot_displace_four_step_leader_from_confirmation(self):
        from superagent.confirmation import shortlist
        rows = [dict(candidate=key,status='completed',primary_metric='accuracy',search_score=s)
                for key,s in zip('dcba',(.6,.5,.8,.8))]
        incumbent=shortlist(rows,1)[0]['candidate']
        for challenger_score in (.4,.8,.9):
            challenger=dict(rows[0],candidate=CANDIDATE,search_score=challenger_score)
            self.assertIn(incumbent,[r['candidate'] for r in shortlist(rows+[challenger],2)])
