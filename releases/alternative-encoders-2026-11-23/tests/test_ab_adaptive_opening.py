import os
import unittest
from unittest.mock import patch

import numpy as np

from superagent import ab_adaptive_opening as policy
from superagent.ab_methods import additions, PREFERENCE_CATALOG_ENV, TEXT_FAMILIARITY, TRANSFER_METHODS
from superagent.metrics import MetricSpec


class AdaptiveOpeningTests(unittest.TestCase):
    def test_uncertainty_uses_independent_groups_and_actual_disagreement(self):
        y = np.zeros(8, dtype=int)
        groups = np.repeat(range(4), 2)
        def measure(a, b):
            return policy.opening_uncertainty({'a':(np.array(a),), 'b':(np.array(b),)}, y, groups)
        self.assertFalse(measure(y, y)['ambiguous'])
        self.assertFalse(measure(y, np.ones(8))['ambiguous'])
        self.assertTrue(measure([0,0,0,0,1,1,0,0], [1,1,0,0,0,0,0,0])['ambiguous'])

    def test_pair_that_excludes_leader_must_confirm_against_it(self):
        y = np.zeros(10, dtype=int)
        tie = np.zeros(10, dtype=bool)
        p = {'leader':[.10]*8+[.99]*2, 'one':[.55]*3+[.05]*7,
             'two':[.05]*3+[.55]*3+[.05]*4, 'same':[.10]*8+[.99]*2}
        columns = {n:((np.array(v)>.5).astype(int), np.array(v), i+1) for i,(n,v) in enumerate(p.items())}
        state = dict(history=[],prediction_evidence=[],task={'metric':MetricSpec('accuracy',(0,1)).record()})
        trusted = {'primary':{'groups':np.arange(10)}}
        second = {}
        def split(_state, _trusted, which, _metric):
            return y, tie, columns if which=='primary' else second
        with patch.object(policy,'selection_evidence'), patch.object(policy.ab_complement,'split_columns',side_effect=split):
            first = policy.review(state,trusted)
            self.assertEqual(first['pending'], ['one','two','leader'])
            second.update({k:columns[k] for k in ('one','two')})
            self.assertEqual(policy.review(state,trusted)['pending'], ['leader'])
            second['leader'] = columns['leader']
            self.assertEqual(policy.review(state,trusted)['members'], ['one','two'])
            second['leader'] = (y, np.full(10,.05), 9)
            self.assertEqual(policy.review(state,trusted)['members'], ['leader'])

    def test_budget_and_feedback_determine_whether_to_explore_a_fifth_method(self):
        review = dict(opening_uncertainty={'ambiguous':True}, evidence_trials=[1,2,3,4])
        state = dict(history=[{}]*4, budget={'remaining_fit_allowance':6})
        with patch.object(policy,'review',return_value=review), patch.object(policy,'next_action',
                side_effect=lambda *a,**k:dict(action='explore',candidate_id='untried')) as choose:
            self.assertEqual(policy.action(state,{}, {'primary':{}})['candidate_id'], 'untried')
            state['budget']['remaining_fit_allowance'] = 5
            self.assertEqual(policy.action(state,{}, {})['action'], 'stop')
            state['budget']['remaining_fit_allowance'] = 6
            review['opening_uncertainty']['ambiguous'] = False
            self.assertEqual(policy.action(state,{}, {})['action'], 'stop')
            choose.assert_called_once()

    def test_joint_menu_is_the_union_of_both_available_preference_routes(self):
        def options(flavor, kind='pair_text'):
            with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:flavor}):
                return {c.family for c in additions(kind)}
        with patch('superagent.ab_sentence.verified_directory',return_value=True), patch(
                'superagent.ab_sentence.verified_files',return_value=True):
            joint = options('deep-e5-familiarity')
            self.assertEqual(joint, options('deep-e5') | options('deep-familiarity'))
            self.assertTrue({TEXT_FAMILIARITY, TRANSFER_METHODS['deep-e5'][1]} <= joint)
            self.assertNotIn(TEXT_FAMILIARITY, options('deep-e5-familiarity','single_text'))
