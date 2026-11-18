import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from superagent.benchmark import run_one
from superagent.data import DevelopmentTask
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM,RunAllowance
from superagent.runtime_budget import RuntimeBudget,PlanningTimeExhausted


class RuntimeBudgetTests(unittest.TestCase):
    def test_uses_worker_wall_cost_and_never_uses_validation_score(self):
        row={'candidate':'a','status':'completed','search_accuracy':.6,'training_seconds':.01,'worker_wall_seconds':4.}
        budget=RuntimeBudget(0,100,60,3,3,True,True,'reserve',clock=lambda:10.)
        first=budget.search_window([row]);second=budget.search_window([{**row,'official_validation_accuracy':1.}])
        self.assertEqual(first,second);self.assertEqual(first['next_fit_estimate_seconds'],6.)
        self.assertEqual(first['refit_reserve_seconds'],12.)
        self.assertTrue(first['allowed'])

    def test_remaining_time_and_planning_latency_can_stop_a_new_request(self):
        clock=[5.];row={'candidate':'a','status':'completed','search_accuracy':.5,'worker_wall_seconds':2.}
        budget=RuntimeBudget(0,30,60,3,3,True,True,'reserve',clock=lambda:clock[0])
        self.assertTrue(budget.search_window([row],False)['allowed'])
        self.assertFalse(budget.search_window([row],True)['allowed'])
        clock[0]=29.;self.assertFalse(budget.search_window([row],False)['allowed'])

    def test_initial_failed_attempts_have_bounded_windows_and_invalid_budget_rejected(self):
        budget=RuntimeBudget(0,10,60,3,3,True,True,'reserve',clock=lambda:0.)
        window=budget.search_window([]);self.assertEqual(window['fit_timeout_seconds'],1.9)
        self.assertEqual(window,budget.search_window([{'candidate':'a','status':'timeout'}]))
        for total in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):RuntimeBudget(0,total,60,3,3,True,True)

    def virtual_pipeline(self,root,mode,confirmation_duration=2.):
        clock=[0.];x=np.arange(240,dtype=float).reshape(80,3);y=np.arange(80)%2
        task=DevelopmentTask('numeric_tabular','fixture',x,y,x[:10],y[:10],{})
        directory=root/mode
        def train(payload,candidate,destination,timeout):
            value=joblib.load(payload)
            duration=confirmation_duration if destination.name.startswith('confirmation_') else 2.
            if duration>timeout:
                clock[0]+=timeout
                return None,{'status':'timeout','training_seconds':timeout,'worker_wall_seconds':timeout}
            clock[0]+=duration
            model=DummyClassifier(strategy='most_frequent').fit(value['X_train'],value['y_train'])
            metrics={'status':'completed','search_accuracy':.5,'training_seconds':.01,'worker_wall_seconds':duration}
            joblib.dump({'model':model,'metrics':metrics},destination)
            return model,metrics
        with patch('superagent.benchmark.time.monotonic',side_effect=lambda:clock[0]),\
             patch('superagent.benchmark.train_candidate',side_effect=train):
            result=run_one(task,'local_balanced_blind',707,directory,None,max_trials=7,time_limit=20,
                trial_timeout=10,anchor_first=True,confirmation_repeats=3,refit_full_train=True,time_policy=mode)
        return result,directory

    def test_same_virtual_time_budget_can_preserve_complete_final_model(self):
        with tempfile.TemporaryDirectory() as temp:
            shared,_=self.virtual_pipeline(Path(temp),'shared')
            reserve,path=self.virtual_pipeline(Path(temp),'reserve')
            self.assertEqual(shared['trials_attempted'],7)
            self.assertFalse(shared['confirmation']['valid_confirmation']);self.assertFalse(shared['refit_succeeded'])
            self.assertLess(reserve['trials_attempted'],7)
            self.assertTrue(reserve['confirmation']['valid_confirmation']);self.assertTrue(reserve['refit_succeeded'])
            self.assertFalse(reserve['runtime_budget']['hard_total_wall_clock_guarantee'])
            self.assertEqual(reserve['runtime_budget']['search_stop_reason'],'reserve_time_for_confirmation_and_refit')
            final=json.loads((path/'final_selection.json').read_text());self.assertEqual(final['training_rows'],80)

    def test_slow_confirmation_is_partial_and_primary_winner_still_gets_a_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            result,path=self.virtual_pipeline(Path(temp),'reserve',confirmation_duration=8.)
            self.assertFalse(result['confirmation']['valid_confirmation'])
            self.assertEqual(result['selected_candidate'],result['primary_best_candidate'])
            self.assertTrue(result['refit_succeeded'])
            self.assertTrue((path/'selected_model.joblib').exists())


class PlanningDeadlineTests(unittest.TestCase):
    def test_expired_window_starts_no_reservation_or_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);key=root/'dummy';key.write_text('SYNTHETIC')
            account=Mock();client=ExperimentLLM('https://example.invalid',key,'DeepSeek-V4.1-Flash',.1,account=account)
            RunAllowance(client,.1).set_deadline(1.)
            logger=RunLogger(root);logger.initialize_usage_ledger()
            with patch('superagent.experiment_llm.time.monotonic',return_value=2.),\
                 patch('superagent.experiment_llm.requests.post') as post:
                with self.assertRaises(PlanningTimeExhausted):client.ask({},logger,'planner')
                post.assert_not_called();account.reserve.assert_not_called()
            self.assertEqual(client.request_count,0);self.assertFalse(client.blocked)

    def test_phase_timeout_fits_remaining_window_and_valid_usage_still_billed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);key=root/'dummy';key.write_text('SYNTHETIC')
            client=ExperimentLLM('https://example.invalid',key,'DeepSeek-V4.1-Flash',.1);client.set_deadline(10.)
            logger=RunLogger(root);logger.initialize_usage_ledger()
            response={'model':'DeepSeek-V4.1-Flash','usage':{'prompt_tokens':100,'completion_tokens':20},
                'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}
            with patch('superagent.experiment_llm.time.monotonic',return_value=6.),\
                 patch('superagent.experiment_llm.requests.post') as post:
                post.return_value.ok=True;post.return_value.json.return_value=response
                self.assertEqual(client.ask({},logger,'planner'),{})
                self.assertEqual(post.call_args.kwargs['timeout'],(1.,3.))
            self.assertAlmostEqual(client.spent,.00018)
            client.set_deadline(None);self.assertIsNone(client.deadline)


if __name__=='__main__':unittest.main()
