from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from superagent import ab_single_conflict as decision
from superagent.metrics import MetricSpec


class SingleConflictTests(unittest.TestCase):
    def test_repeated_rows_receive_one_vote_and_seen_groups_do_not_decide_the_third_split(self):
        stats = decision.reconcile([
            ([1,2],['g','a'],[1,1],[1,1],[0,1]),
            ([1,3],['g','b'],[1,1],[0,0],[1,1]),
            ([4,5],['g','c'],[1,1],[0,1],[1,1])])
        self.assertEqual((stats['unique_rows'],stats['row_observations']),(5,6))
        self.assertEqual((stats['fresh_rows'],stats['fresh_net_correct']),(1,0))
        self.assertAlmostEqual(stats['unique_gain'],.4)
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            decision.reconcile([([1],['a'],[1],[1],[0]),([1],['a'],[0],[0],[1])])

    def evidence(self, *, third=False, fresh=True, flip=False, budget=4):
        anchor,rival=('b','a') if flip else ('a','b')
        base=dict(anchor=anchor,members=[anchor],promoted=False,opening_correct={anchor:3,rival:2},
                  confirmation_correct={anchor:1,rival:3},pending=[],evidence_trials=[1,2,3,4])
        state=dict(task={'metric':MetricSpec('accuracy',(0,1)).record()},registered_confirmation_splits=['confirmation_1','confirmation_2'],
                   budget={'remaining_fit_allowance':budget,'reserved_final_refit':2})
        names=['primary','confirmation_1','confirmation_2']
        trusted={n:{'row_ids':list(range(i*3,i*3+3)),'groups':[str(i if fresh else 0)]*3} for i,n in enumerate(names)}
        ys=np.ones(3,dtype=int)
        def columns(_state, _trusted, split, _metric):
            i=names.index(split)
            left,right=([1,1,1],[1,1,0]) if i==0 else ([1,0,0],[1,1,1])
            cols={anchor:(np.array(left),ys,2*i+1), rival:(np.array(right),ys,2*i+2)}
            return ys,ys,{} if i==2 and not third else cols
        stop={'action':'stop'}
        with patch.object(decision.ab_adaptive_opening,'review',side_effect=lambda *args:deepcopy(base)), \
             patch.object(decision.ab_adaptive_opening,'action',return_value=stop), \
             patch.object(decision.ab_complement,'split_columns',side_effect=columns):
            return decision.review(state,trusted),decision.action(state,{},trusted)

    def test_complete_block_is_admitted_only_when_affordable(self):
        review,action=self.evidence()
        self.assertEqual(review['pending'],['a','b'])
        self.assertEqual((action['action'],action['split_id'],action['candidate_id']),('confirm','confirmation_2','a'))
        review,action=self.evidence(budget=3)
        self.assertEqual(review['single_conflict']['status'],'retain_budget')
        self.assertEqual(action['action'],'stop')

    def test_verified_third_result_can_replace_either_named_anchor(self):
        for flip,expect in [(False,'b'),(True,'a')]:
            review,action=self.evidence(third=True,flip=flip)
            self.assertEqual((review['single_conflict']['status'],review['members'],action['action']),('switch',[expect],'stop'))
            self.assertEqual(review['evidence_trials'],[1,2,3,4,5,6])
        review,_=self.evidence(third=True,fresh=False)
        self.assertEqual((review['single_conflict']['status'],review['members']),('retain',['a']))

    def test_no_challenge_after_pair_promotion_or_agreement(self):
        for result in ({'promoted':True,'members':['a','b']},
                       {'promoted':False,'anchor':'a','members':['a'],'opening_correct':{'a':7,'b':5},
                        'confirmation_correct':{'a':6,'b':5}}):
            with patch.object(decision.ab_adaptive_opening,'review',return_value=result):
                self.assertEqual(decision.review({'registered_confirmation_splits':['confirmation_1','confirmation_2']},{}),result)
