from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from superagent.autonomous_run import partitions
from superagent.metrics import MetricSpec
from superagent.remaining_fold_review import admission, assessment_folds


class RemainingFoldTests(TestCase):
    def test_search_rows_are_unchanged_and_each_new_group_is_scored_once(self):
        x=np.repeat(np.arange(800),2)[:,None]
        y=x[:,0]%3
        task=SimpleNamespace(kind="numeric_tabular",X_train=x,y_train=y,spec={})
        initial=partitions(task,19)
        extra=assessment_folds(task,19,initial,partitions)
        self.assertEqual(len(extra),2)
        held=np.concatenate([h for _,h,_ in [*initial.values(),*extra.values()]])
        np.testing.assert_array_equal(np.sort(held),np.arange(len(y)))

    def test_positive_accuracy_must_repeat_on_both_folds(self):
        y=[np.tile([0,1],240)]*2
        old=[(v.copy(),None) for v in y]
        for a,_ in old:
            a[::11]=1-a[::11]
        perfect=[(v,None) for v in y]
        keys=[np.arange(480),np.arange(480,960)]
        metric=MetricSpec("accuracy",[0,1])
        classes=np.array([0,1])
        self.assertTrue(admission(metric,classes,y,old,perfect,keys)[0])
        bad=y[1].copy()
        bad[::6]=1-bad[::6]
        self.assertFalse(admission(metric,classes,y,old,[perfect[0],(bad,None)],keys)[0])
