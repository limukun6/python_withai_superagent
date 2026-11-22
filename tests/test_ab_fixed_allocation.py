from copy import deepcopy
import unittest

from superagent.ab_catalog import capabilities_for, untried_candidates
from superagent.ab_scheduler import next_action
from superagent.scheduler_bridge import validate_action
from tests.test_ab_scheduler import competing


class TestFixedAllocation(unittest.TestCase):
    def test_same_anchor_and_fixed_fourth_explore_while_adaptive_confirms(self):
        state = competing()
        caps = capabilities_for(state['task'], train_rows=256)
        state['available_candidates'] = untried_candidates(caps, state['history'])
        self.assertEqual(next_action(state, capabilities=caps)['action'], 'confirm')
        fixed = next_action(state, capabilities=caps, allocation='fixed')
        self.assertEqual(fixed['action'], 'explore')
        self.assertEqual(validate_action(state, fixed, caps, {'primary': {}})['expected_classifier_fits'], 1)
        state.update(history=[], prediction_evidence=[], available_candidates=untried_candidates(caps, []))
        self.assertEqual(next_action(state, capabilities=caps), next_action(state, allocation='fixed', capabilities=caps))

    def test_fewer_than_four_candidates_confirm_top_three_and_preserve_refit(self):
        state = competing()
        caps = capabilities_for(state['task'], train_rows=256)
        state['available_candidates'] = []
        action = next_action(state, allocation='fixed', capabilities=caps)
        top = sorted(state['history'], key=lambda row: (-row['search_score'], row['candidate']))
        self.assertEqual((action['action'], action['candidate_id'], action['split_id']),
                         ('confirm', top[0]['candidate'], 'confirmation_1'))
        self.assertEqual(set(action['evidence_trials']), {1, 2, 3})
        validate_action(state, action, caps, {'confirmation_1': {}})
        constrained = deepcopy(state)
        constrained['budget'].update(attempted_classifier_fits=8, completed_classifier_fits=8, remaining_fit_allowance=3)
        self.assertEqual(next_action(constrained, allocation='fixed', capabilities=caps)['action'], 'stop')

    def test_unknown_budget_or_policy_cannot_trigger_training(self):
        state = competing()
        state['budget']['unknown_failed_attempts'] = 1
        self.assertEqual(next_action(state, allocation='fixed')['action'], 'stop')
        with self.assertRaisesRegex(ValueError, 'Unknown AB budget allocation'):
            next_action(state, allocation='not-a-policy')
