import unittest
from unittest.mock import patch

from superagent import ab_adaptive_opening as policy


class EncoderCoverageTests(unittest.TestCase):
    def row(self, name, backbone, revision='pin', fits=1):
        return {'id':name, 'resource':'local_frozen_encoder_cpu','params':{'encoder':backbone,'revision':revision},
                'classifier_fits':fits, 'refit_classifier_fits':1, 'supports_probabilities':True}

    def test_unseen_means_a_new_verified_backbone_within_the_registered_atomic_fit_budget(self):
        items=[self.row('r0','backbone-a'),self.row('r1','backbone-a','second-revision'),
               self.row('r2','backbone-b'),self.row('r3','backbone-c',''),self.row('r4','backbone-d',fits=2)]
        caps={r['id']:r for r in items}
        state={'history':[{'candidate':'r0','split_id':'primary'}], 'available_candidates':items[1:]}
        self.assertEqual(policy.unseen_encoder_candidates(state,caps),[items[2]])

    def test_a_clear_leader_can_leave_an_encoder_untested_and_the_budget_still_controls_exploration(self):
        observed=[self.row('seen'+str(i),'a') for i in range(4)]
        fifth=self.row('opaque-5','b')
        caps={r['id']:r for r in [*observed,fifth]}
        state=dict(history=[dict(candidate=r['id'],split_id='primary') for r in observed],
                   available_candidates=[fifth],budget={'remaining_fit_allowance':6})
        receipt=dict(opening_uncertainty={'ambiguous':False},evidence_trials=[1,2,3,4])
        def choose(projected,**_kwargs):
            self.assertEqual(projected['available_candidates'],[fifth])
            return dict(action='explore',candidate_id=fifth['id'])
        with patch.object(policy,'review',return_value=receipt),patch.object(policy,'next_action',side_effect=choose):
            self.assertEqual(policy.action(state,caps,{})['action'],'stop')
            self.assertEqual(policy.action(state,caps,{'primary':{}},cover_untried_encoder=True)['candidate_id'],'opaque-5')
            state['budget']['remaining_fit_allowance']=5
            self.assertEqual(policy.action(state,caps,{},cover_untried_encoder=True)['action'],'stop')
