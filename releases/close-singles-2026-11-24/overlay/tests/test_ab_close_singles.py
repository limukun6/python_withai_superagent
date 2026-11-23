import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np

from superagent import ab_close_singles as policy
from superagent.metrics import MetricSpec


class CloseSinglesTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.ones(8, dtype=int)
        self.opening = self.records({
            'anchor': [1, 1, 1, 1, 1, 0, 0, 0],
            'two_corrections': [1, 1, 0, 0, 1, 1, 0, 1],
            'one_correction': [1, 1, 1, 1, 0, 1, 0, 0],
            'empty': [0] * 8,
        })
        self.confirmed = {}
        self.first_groups = [f'g{i}' for i in range(8)]
        self.trusted = {'primary': {'groups': self.first_groups},
                        'confirmation_1': {'groups': self.first_groups[:4] + [f'h{i}' for i in range(4)]}}
        self.state = dict(task={'metric': MetricSpec('accuracy', (0, 1)).record()},
                          registered_confirmation_splits=['confirmation_1', 'confirmation_2'],
                          history=[{'split_id': 'primary'}] * 4,
                          budget=dict(remaining_fit_allowance=6, reserved_final_refit=2, completed_classifier_fits=4,
                                      attempted_classifier_fits=4, fit_limit=10, unknown_failed_attempts=0,
                                      llm_request_limit=0, used_llm_requests=0, elapsed_seconds=10, time_limit_seconds=1200))
        self.original = dict(anchor='anchor', members=['anchor'], comparison_members=[], evidence_trials=[1, 2, 3, 4],
                             opening_correct={k: int(np.sum(row[0])) for k, row in self.opening.items()})
        self.base = Mock()
        self.base.review.side_effect = lambda *unused: deepcopy(self.original)
        self.base.action.return_value = {'action': 'stop'}
        self.policy = policy.SingleConfirmationPolicy(self.base)
        def columns(state, trusted, split, metric):
            return self.labels, None, self.opening if split == 'primary' else self.confirmed
        context = patch.object(policy, 'split_columns', side_effect=columns)
        context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def records(predictions, start=1):
        return {k: (np.array(v), None, trial) for trial, (k, v) in enumerate(predictions.items(), start)}

    def decision(self):
        return self.policy.action(self.state, {}, self.trusted)

    def review(self):
        return self.policy.review(self.state, self.trusted)

    def test_corrected_errors_choose_the_challenger_after_candidate_renaming(self):
        for alias in ('first', 'zzz'):
            renamed = {alias if k == 'two_corrections' else k: v for k, v in self.opening.items()}
            chosen = policy.close_challenger(renamed, 'anchor', self.labels, self.first_groups)
            self.assertEqual((chosen['challenger'], chosen['corrections']), (alias, 2))
        self.assertIsNone(policy.close_challenger({'anchor': self.opening['anchor'], 'copy': self.opening['anchor']},
                                                 'anchor', self.labels, self.first_groups))

    def test_matching_confirmations_preserve_the_refit_allowance(self):
        self.assertEqual(self.decision()['candidate_id'], 'anchor')
        self.confirmed = self.records({'anchor': [1] * 8}, start=5)
        self.assertEqual(self.decision()['candidate_id'], 'two_corrections')
        self.assertEqual(self.review()['evidence_trials'], [1, 2, 3, 4, 5])
        self.confirmed.clear()
        self.state['budget'].update(remaining_fit_allowance=3, attempted_classifier_fits=7, completed_classifier_fits=7)
        self.assertEqual(self.review()['close_singles']['status'], 'retain_budget')
        self.assertEqual(self.decision()['action'], 'stop')

    def test_switches_when_new_groups_favor_challenger_despite_reused_groups(self):
        self.confirmed = self.records({'anchor': [1, 1, 1, 1, 0, 0, 0, 1],
                                       'two_corrections': [0, 0, 0, 0, 1, 1, 0, 1]}, 5)
        result = self.review()
        self.assertEqual(result['members'], ['two_corrections'])
        self.assertEqual(result['close_singles']['fresh_net'], 2)
        self.assertEqual(result['close_singles']['distinct_rows'], 12)
        self.assertAlmostEqual(result['close_singles']['pooled_gain'], 2 / 12)
        self.assertEqual(self.decision()['action'], 'stop')

    def test_expired_or_uncertain_budget_never_requests_more_confirmations(self):
        for changes in ({'elapsed_seconds': 1200}, {'unknown_failed_attempts': 1},
                        {'remaining_fit_allowance': 7}, {'used_llm_requests': 4}):
            with self.subTest(changes=changes), patch.dict(self.state['budget'], changes):
                self.assertEqual(self.decision()['action'], 'stop')
                self.assertEqual(self.review()['close_singles']['status'], 'retain_budget')
                self.assertTrue(self.review()['close_singles']['budget_reason'])

    def test_gains_only_on_repeated_groups_cannot_replace_the_original(self):
        self.confirmed = self.records({'anchor': [0, 0, 0, 0, 1, 1, 1, 0],
                                       'two_corrections': [1, 1, 1, 1, 1, 1, 1, 0]}, 5)
        self.assertEqual(self.review()['members'], ['anchor'])
        self.assertEqual(self.review()['close_singles']['status'], 'retain')

    def test_requires_fresh_groups_a_registered_split_and_an_affordable_pair(self):
        self.trusted['confirmation_1']['groups'] = self.first_groups
        self.assertEqual(self.review()['close_singles']['status'], 'no_fresh_groups')
        self.state['registered_confirmation_splits'] = []
        self.assertIs(self.decision(), self.base.action.return_value)

    def test_existing_ensemble_or_exploration_has_precedence(self):
        self.original['comparison_members'] = ['anchor', 'two_corrections']
        self.assertIs(self.decision(), self.base.action.return_value)
        self.base.action.return_value = {'action': 'explore'}
        self.base.review.reset_mock()
        self.assertIs(self.decision(), self.base.action.return_value)
        self.base.review.assert_not_called()
