import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from superagent.conditional_review import PASS, FAIL, feasible_probes, group_observation, review_delivery, validate
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.task_budget import TaskBudgetLedger


def observed(fold, correct):
    labels=np.arange(10)%2
    prediction=labels.copy()
    prediction[correct:]=1-prediction[correct:]
    return dict(labels=labels,predictions=prediction,probabilities=None,row_key=fold,score=correct/10.)


def setup():
    names=('primary','confirmation_1','confirmation_2')
    history={f:{'a':observed(f,7)} for f in names}
    history[names[0]]['b']=observed(names[0],8)
    history[names[1]]['b']=observed(names[1],6)
    folds={f:([],np.arange(10)+10*i,{}) for i,f in enumerate(names)}
    caps={n:dict(classifier_fits=1,refit_classifier_fits=1,estimator='tree',representation='raw') for n in ('a','b')}
    return history,folds,caps,MetricSpec('accuracy',(0,1))


def proposal(**extra):
    values=dict(decision='measure',option=0,cited_folds=['primary','confirmation_1'],if_pass=PASS,if_fail=FAIL,
                minimum_delta=None,reason='The two relative scores have opposite signs.',opportunity_cost='One fit and a possible refit.')
    return dict(values,**extra)


class ConditionalReviewTests(unittest.TestCase):
    def test_only_disjoint_affordable_effective_forks_can_interrupt(self):
        hist,folds,caps,metric=setup()
        def probes(remaining=3):return feasible_probes(['a'],hist,folds,caps,metric,remaining)[0]
        self.assertEqual(len(probes()),1)
        self.assertEqual(probes()[0]['measurement_fits'],1)
        self.assertAlmostEqual(probes()[0]['required_delta'],.009)
        self.assertFalse(probes(1))
        hist['confirmation_2']['b']=observed('confirmation_2',8)
        self.assertFalse(probes())
        del hist['confirmation_2']['b']
        hist['confirmation_2']['a']=observed('confirmation_2',10)
        self.assertFalse(probes())
        hist['confirmation_2']['a']=observed('confirmation_2',7)
        folds['confirmation_2']=folds['primary']
        self.assertFalse(probes())

    def test_identical_impossible_or_ungrounded_plans_are_refused(self):
        hist,folds,caps,metric=setup()
        opts,_=feasible_probes(['a'],hist,folds,caps,metric,3)
        for kwargs in [dict(if_pass=FAIL),dict(if_fail=PASS),dict(minimum_delta=.9),dict(minimum_delta=float('nan')),
                       dict(cited_folds=['primary','unseen']),dict(option=2),dict(opportunity_cost='')]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                validate(proposal(**kwargs),dict(options=opts))

    def test_measured_result_executes_precommitted_branch_and_keeps_saved_model_on_error(self):
        for correct,reply,status,action in [(9,proposal(),'adopted',PASS),(6,proposal(),'measured',FAIL),
                (9,proposal(if_pass=FAIL),'retained_after_error',FAIL),(9,dict(decision='keep',reason='Too little to gain'),'declined',FAIL)]:
            with self.subTest(status=status),TemporaryDirectory() as tmp:
                folder=Path(tmp)
                hist,folds,caps,metric=setup()
                executor=Mock(limit=16,caps=caps)
                executor.accounting.return_value=dict(attempted_classifier_fits=4,unknown_failed_attempts=0)
                def fit(*args,**kwargs):
                    log=json.loads((folder/'conditional_review.json').read_text())
                    self.assertEqual(log['status'],'plan_frozen')
                    self.assertNotEqual(log['plan']['if_pass'],log['plan']['if_fail'])
                    self.assertEqual(kwargs['reserve'],1)
                    return object(),dict()
                executor.worker.side_effect=fit
                client=Mock()
                client.ask.return_value=reply
                saved=dict(model_sha256='original')
                new=dict(model_sha256='new')
                with patch('superagent.conditional_review.joblib.load'), \
                        patch('superagent.autonomous_run.planner_observation',return_value=observed('confirmation_2',correct)), \
                        patch('superagent.autonomous_run.freeze_model',return_value=new) as refit:
                    final,target=review_delivery(None,{},metric,saved,dict(members=['a']),executor,folder,RunLogger(folder),20,
                        hist,{'confirmation_2':dict(path='payload')},folds,client,{})
                receipt=json.loads((folder/'conditional_review.json').read_text())
                self.assertEqual((receipt['status'],receipt['executed']),(status,action))
                self.assertEqual(refit.called,action==PASS)
                self.assertEqual(target['members'],['b' if action==PASS else 'a'])
                self.assertEqual(receipt['delivered_sha256'],final['model_sha256'])
                self.assertEqual(executor.worker.called,status in ('adopted','measured'))

    def test_no_conflict_no_call_and_missing_reference_costs_a_real_fit(self):
        hist,folds,caps,metric=setup()
        del hist['confirmation_2']['a']
        self.assertFalse(feasible_probes(['a'],hist,folds,caps,metric,2)[0])
        self.assertEqual(set(feasible_probes(['a'],hist,folds,caps,metric,3)[0][0]['missing']),{'a','b'})
        hist['confirmation_1']['b']=observed('confirmation_1',9)
        with TemporaryDirectory() as tmp:
            client=Mock()
            executor=Mock(caps=caps,limit=16)
            executor.accounting.return_value=dict(attempted_classifier_fits=8,unknown_failed_attempts=0)
            review_delivery(None,{},metric,dict(model_sha256='a'),dict(members=['a']),executor,Path(tmp),RunLogger(Path(tmp)),
                            20,hist,{},folds,client,{})
            client.ask.assert_not_called()
            executor.worker.assert_not_called()

    def test_shared_ensemble_member_is_measured_once_and_auc_uses_scores(self):
        hist,folds,caps,metric=setup()
        caps['c']=caps['a'].copy()
        for bank in hist.values():
            bank['c']=dict(bank['a'])
            for row in bank.values():
                row['probabilities']=np.eye(2)[row['predictions']]
        del hist['confirmation_2']['c']
        options,_=feasible_probes(['a','c'],hist,folds,caps,metric,3)
        self.assertEqual([o['challenger'] for o in options],['b'])
        self.assertEqual(options[0]['missing'],['c','b'])
        self.assertEqual(options[0]['measurement_fits'],2)
        labels=np.array([0,0,1,1])
        probability=np.array([[.4,.6],[.3,.7],[.2,.8],[.1,.9]])
        row=dict(labels=labels,predictions=np.ones(4),probabilities=probability,row_key='paired-rows')
        combined=group_observation({'one':row,'two':row},['one','two'],MetricSpec('roc_auc',(0,1)))
        self.assertEqual(combined['score'],1.)
        self.assertEqual(np.mean(combined['predictions']==labels),.5)

    def test_paired_calls_share_one_scope_and_cannot_be_replayed(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'local.sqlite'
            ledger=TaskBudgetLedger.initialize(path,new_project=True)
            model='DeepSeek-V4.1-Flash'
            with sqlite3.connect(path) as db:
                db.executemany('INSERT INTO metadata VALUES (?,?)',[('ledger_kind','provisional_local_outbox'),
                                                                   ('team_merge_state','pending_team_handoff')])
            seed=ledger.reserve('past',.01,'past',model)
            ledger.settle(seed,model,10,10)
            with self.assertRaises(ValueError):
                ledger.preflight('past',.12,paired_arm='flash')
            first=ledger.reserve('paired',.12,'run-A',model,paired_arm='flash')
            with self.assertRaises(ValueError):
                ledger.preflight('paired',.12,paired_arm='conditional-flash')
            ledger.settle(first,model,100,50)
            with self.assertRaises(ValueError):
                ledger.reserve('paired',.12,'another-directory',model,paired_arm='flash')
            second=ledger.reserve('paired',.12,'run-B',model,paired_arm='conditional-flash')
            ledger.settle(second,model,100,50)
            self.assertAlmostEqual(ledger.snapshot('paired')['known_task_cost_rmb'],.0006)
            with self.assertRaises(ValueError):
                ledger.reserve('paired',.12,'rerun',model,paired_arm='conditional-flash')
