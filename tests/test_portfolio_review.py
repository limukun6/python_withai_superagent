import unittest

import numpy as np

from superagent.autonomous_run import STANDARD_FOLDS, partitions
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec
from superagent.portfolio_review import EXTRA_FOLDS, nominate, reserve_fits


class PortfolioReview(unittest.TestCase):
    @staticmethod
    def observation(metric,truth,pred,confidence=.9):
        order=list(metric.labels)
        matrix=np.full((len(truth),len(order)),(1-confidence)/(len(order)-1))
        for i,value in enumerate(pred):
            matrix[i,order.index(value)]=confidence
        return dict(probabilities=matrix,labels=np.asarray(truth),predictions=np.asarray(pred),row_key="primary")

    def test_nomination_uses_declared_metric_and_is_invariant_to_aliases(self):
        # A rare-class gain outweighs a small majority-class loss only under macro F1.
        truth=[0]*8+[1,2]
        pred_a=[0]*10
        pred_b=[0]*5+[1]*3+[1,2]
        for name,expected in (("accuracy",False),("macro_f1",True)):
            metric=MetricSpec(name,(2,0,1))
            for a,b in (("a","b"),("anonymous_22","anonymous_01")):
                primary={a:self.observation(metric,truth,pred_a,.95),b:self.observation(metric,truth,pred_b,.85)}
                caps={k:dict(params={"C":v},classifier_fits=1,refit_classifier_fits=1) for k,v in ((a,.1),(b,.5))}
                _,_,choice=nominate(primary,[a],[1.],caps,7,metric,(),set())
                self.assertEqual(choice is not None,expected)
                if expected:
                    self.assertEqual(choice["member"],b)
                self.assertIsNone(nominate(primary,[a],[1.],caps,4,metric,(),set())[2])
                self.assertIsNone(nominate(primary,[a],[1.],caps,7,metric,(),{b})[2])

    def test_misaligned_rows_are_rejected(self):
        metric=MetricSpec("accuracy",(1,0))
        rows=self.observation(metric,[0,1],[0,1])
        caps={k:dict(params={"C":1},classifier_fits=1,refit_classifier_fits=1) for k in ("one","two")}
        with self.assertRaisesRegex(ValueError,"ordered"):
            nominate({"one":rows,"two":dict(rows,row_key="other")},["one"],[1.],caps,10,metric,(),set())

    def test_existing_three_folds_and_budget_remain_reproducible(self):
        rows=np.arange(150,dtype=float).reshape(75,2)
        task=TrainingTask("numeric_tabular",rows,np.arange(75)%3,{"evaluation_protocol":{"kind":"grouped_stratified"}})
        base=partitions(task,83)
        all_folds=partitions(task,83,names=(*STANDARD_FOLDS,*EXTRA_FOLDS))
        self.assertEqual(len(all_folds),5)
        for k,(fit,held,_) in base.items():
            np.testing.assert_array_equal(all_folds[k][0],fit)
            np.testing.assert_array_equal(all_folds[k][1],held)
        self.assertEqual(len(set(np.concatenate([v[1] for v in all_folds.values()]))),len(rows))
        self.assertEqual(reserve_fits(22,3,other_reserve=6),0)
        self.assertEqual(reserve_fits(32,3,other_reserve=6),10)
