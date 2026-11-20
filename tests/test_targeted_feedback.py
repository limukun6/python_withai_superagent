import copy
import unittest

from superagent.contracts import ExperimentProposal
from superagent.hypothesis_agent import check_expectation
from superagent.targeted_feedback import candidate_facts
from superagent.contracts import DecisionState
from superagent.checked_history_policy import decision_context


class TargetedFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.slice = dict(slice_id='pair_token_overlap_high', feature='pair_token_overlap',
                          threshold_from_training=.5, relation='high', rows=100, accuracy=.6)
        self.reference = dict(trial=1, candidate='char_C1.0', status='completed', search_accuracy=.65,
                              error_slices={'slices': [self.slice]})
        self.proposal = dict(candidate='char_C0.3',
            expected_change=dict(metric='search_accuracy', reference_trial=1,
                                 direction='increase', min_delta=.01),
            slice_expectation=dict(slice_id='pair_token_overlap_high', reference_trial=1, min_delta=.01))

    def test_overall_gain_cannot_hide_unchanged_target(self):
        result = copy.deepcopy(self.reference)
        result['search_accuracy'] = .7
        check = check_expectation(self.proposal, result, [self.reference])
        self.assertEqual(check['overall_check']['status'], 'supported')
        self.assertEqual(check['slice_check']['status'], 'not_supported')
        self.assertEqual(check['status'], 'not_supported')

    def test_changed_definition_is_not_comparable(self):
        result = copy.deepcopy(self.reference)
        result['error_slices']['slices'][0]['threshold_from_training'] = .7
        self.assertEqual(check_expectation(self.proposal, result, [self.reference])['slice_check']['status'],
                         'not_evaluable')

    def test_wrong_classifier_and_future_slice_are_rejected(self):
        config = dict(id='char_C0.3', params={'C': .3})
        raw = dict(candidate='char_C0.3', task_kind='pair_text', configuration_claim={'classifier': 'linear_svm'})
        with self.assertRaisesRegex(ValueError, 'classifier'):
            ExperimentProposal.from_decision(2, config, 'test', raw, [self.reference])
        raw = self.proposal | dict(slice_expectation=self.proposal['slice_expectation'] | {'reference_trial': 2})
        with self.assertRaisesRegex(ValueError, 'observed prior slice'):
            ExperimentProposal.from_decision(2, config, 'test', raw, [self.reference])

    def test_facts_match_executable_candidate(self):
        self.assertEqual(candidate_facts('pair_text', {'id': 'char_C0.3', 'params': {'C': .3}})['classifier'],
                         'logistic')

    def test_raw_condition_cannot_see_derived_checks(self):
        state = DecisionState(dict(history=[self.reference | {'hypothesis_check': {'status': 'not_supported'}}],
            task_spec={'kind': 'numeric_tabular'}, available_candidates=[], trial_budget={'remaining': 1},
            memory={'hypotheses': [{'status': 'measurement_not_supported'}]}))
        raw = decision_context(state, False)
        checked = decision_context(state, True)
        self.assertNotIn('hypothesis_check', raw['history'][0])
        self.assertNotIn('memory', raw)
        self.assertEqual(checked.pop('checks')[0]['hypothesis_check']['status'], 'not_supported')
        checked.pop('memory')
        self.assertEqual(raw, checked)

    def test_generic_numeric_classifier_fact(self):
        row = candidate_facts('numeric_tabular', {'id': 'numeric_tabular_extra_trees_leaf1', 'params': {}})
        self.assertEqual(row['classifier'], 'extra_trees')

    def test_wall_clock_never_reaches_the_model(self):
        state = DecisionState(dict(history=[self.reference | {'training_seconds': 12.3456789}],
            task_spec={'kind': 'pair_text'}, available_candidates=[], trial_budget={'remaining': 1},
            memory=None))
        for checked in (False, True):
            row = decision_context(state, checked)['history'][0]
            self.assertNotIn('training_seconds', row)
            self.assertEqual(row['candidate'], 'char_C1.0')


if __name__ == '__main__':
    unittest.main()
