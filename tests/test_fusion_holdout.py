import unittest
import numpy as np
from superagent.fusion_holdout import compare_scores
from superagent.metrics import MetricSpec


class FusionHoldoutTests(unittest.TestCase):
    def test_only_consistent_new_evidence_releases_the_vote(self):
        y=np.arange(240)%2
        groups=np.repeat(np.arange(120),2)
        labels=np.array([0,1])
        metric=MetricSpec("accuracy",[0,1])
        p=np.eye(2)[y].astype(float)
        old=p.copy()
        old[::8]=old[::8,::-1]
        yes,checks=compare_scores(metric,y,labels,old,p,groups)
        self.assertTrue(yes)
        self.assertEqual(sum(r["rows"] for r in checks[1:]),len(y))
        self.assertFalse(compare_scores(metric,y,labels,p,old,groups)[0])
        self.assertFalse(compare_scores(metric,y,labels,p,p,groups)[0])
