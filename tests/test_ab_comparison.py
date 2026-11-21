import unittest

import numpy as np
from sklearn.metrics import f1_score

from superagent.ab_comparison import incumbent_review, macro_f1_pair_spread


class TestPairedReview(unittest.TestCase):
    def review(self, rows):
        measured, records = {}, {}
        for trial,(name,split,score) in enumerate(rows,1):
            measured[name,split] = dict(trial=trial,candidate=name,split_id=split,status='completed',search_score=score)
            records[trial] = [0,1]
        return incumbent_review(measured,records,['confirmation_1','confirmation_2'])

    def test_an_unreplicated_new_leader_cannot_replace_an_incumbent(self):
        p,a,b = 'primary','confirmation_1','confirmation_2'
        rows = [('first',p,.65),('second',p,.60),('third',p,.50),('first',a,.60),('second',a,.70)]
        self.assertEqual(self.review(rows)['incumbent'],'first')
        rows += [('first',b,.62),('second',b,.68),('new',p,.99)]
        result = self.review(rows)
        self.assertEqual(result['incumbent'],'second')
        self.assertEqual(result['promotions'][0]['evidence_trials'],[1,2,4,5,6,7])
        self.assertEqual(result['basis'],[p,a,b])
        rows += [('new',a,.98)]
        self.assertEqual(self.review(rows)['incumbent'],'new')

    def test_threshold_and_missing_comparable_partition(self):
        p,a,b = 'primary','confirmation_1','confirmation_2'
        rows = [('first',p,.60),('second',p,.55),('old',p,.40),('new',p,.601)]
        self.assertEqual(self.review(rows+[('new',a,.99),('first',b,.45)])['incumbent'],'first')
        self.assertEqual(self.review(rows+[('new',a,.701),('first',a,.70)])['incumbent'],'first')

    def test_macro_f1_group_resampling_matches_direct_common_index_calculation(self):
        y = np.array([0,0,1,1,1,2,2,2])
        a = np.array([0,0,1,0,2,2,1,2])
        b = np.array([1,0,2,1,1,2,0,2])
        group = np.repeat(np.arange(4),2)
        spread = macro_f1_pair_spread(y,a,b,[0,1,2],group)
        rng = np.random.default_rng(902)
        observed = []
        for _ in range(128):
            idx = np.concatenate([np.flatnonzero(group==g) for g in rng.integers(4,size=4)])
            vals = [f1_score(y[idx],v[idx],labels=[0,1,2],average='macro',zero_division=0) for v in (a,b)]
            observed.append(vals[0]-vals[1])
        self.assertAlmostEqual(spread,np.std(observed,ddof=1))
        self.assertEqual(macro_f1_pair_spread(y,a,a,[0,1,2],group),0.)
