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
