import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
import joblib

import numpy as np

from superagent.data import DevelopmentTask
from superagent.models import DummyClassifier, search_candidates
from superagent.task_semantics import (SWAP_TOLERANCE, declared_semantics, expectations_for,
                                       probe_model_swap, swap_augmented, swap_behavior,
                                       validate_task_semantics)


def pair_task(kind='pair_text', semantics=None):
    schema = {'modality': 'pair_text', 'pair_label_one': 'left'}
    if semantics is not None:
        schema['pair_semantics'] = semantics
    return DevelopmentTask(kind, 'unit.zip', [('a', 'b')], np.asarray([1]), None, None,
                           {'input_schema': schema})


class PairSemanticsTests(unittest.TestCase):
    def test_undeclared_semantics_is_recorded_not_rejected(self):
        record = validate_task_semantics(pair_task())
        self.assertIsNone(declared_semantics(pair_task()))
        self.assertEqual(record['status'], 'undeclared')
        self.assertEqual(record['swap_policy'], 'unverified')
        self.assertEqual(record['excluded_candidates'], [])
        self.assertIn('semantics_declared', [c['check'] for c in record['checks']])

    def test_preference_keeps_swapping_augmentation(self):
        record = validate_task_semantics(pair_task(semantics='preference'))
        self.assertEqual(record['status'], 'accepted')
        self.assertEqual(record['swap_policy'], 'flips')
        self.assertEqual(record['excluded_candidates'], [])
        augmented = swap_augmented([{'id': c.family, 'params': c.params}
                                    for c in search_candidates('pair_text', 'expanded')])
        self.assertTrue(augmented, 'the expanded pair portfolio should contain flipping augmentation')

    def test_symmetric_semantics_removes_label_flipping_candidates(self):
        record = validate_task_semantics(pair_task(semantics='symmetric'), portfolio='semantic')
        augmented = swap_augmented([{'id': c.family, 'params': c.params}
                                    for c in search_candidates('pair_text', 'semantic')])
        self.assertEqual(record['status'], 'rejected')
        self.assertEqual(record['swap_policy'], 'preserves')
        self.assertTrue(set(augmented) <= set(record['excluded_candidates']))
        self.assertEqual(record['remaining_candidates'], 0)
        recipes = [c.family for c in search_candidates('pair_text','semantic') if c.family.startswith('recipe_')]
        self.assertEqual(len(recipes),4)
        self.assertTrue(set(recipes) <= set(record['excluded_candidates']))
        self.assertEqual([c['status'] for c in record['checks'] if c['check'] == 'swap_augmentation'],
                         ['filtered'])

    def test_default_pair_portfolio_cannot_serve_a_symmetric_task(self):
        record = validate_task_semantics(pair_task(semantics='symmetric'))
        every_candidate = swap_augmented([{'id': c.family, 'params': c.params}
                                          for c in search_candidates('pair_text', 'expanded')])
        self.assertEqual(record['status'], 'rejected')
        self.assertEqual(record['remaining_candidates'], 0)
        self.assertEqual(len(record['excluded_candidates']), len(every_candidate))
        self.assertTrue(any('nothing can be run safely' in reason for reason in record['reasons']))

    def test_unknown_semantics_is_rejected_with_a_reason(self):
        record = validate_task_semantics(pair_task(semantics='similarity'))
        self.assertEqual(record['status'], 'rejected')
        self.assertTrue(any('unknown pair semantics' in reason for reason in record['reasons']))

    def test_relation_must_load_as_the_multiclass_form(self):
        record = validate_task_semantics(pair_task(semantics='relation'))
        self.assertEqual(record['status'], 'rejected')
        self.assertTrue(any('text_tabular' in reason for reason in record['reasons']))

    def test_expected_pattern_follows_the_declared_policy(self):
        self.assertEqual(expectations_for('preference')['expected_pattern'], 'flips')
        self.assertEqual(expectations_for('symmetric')['expected_pattern'], 'preserves')
        self.assertIsNone(expectations_for('relation')['expected_pattern'])


class SwapBehaviorTests(unittest.TestCase):
    def test_near_chance_predictions_cannot_distinguish_two_rules(self):
        for value in (.5,.52):
            self.assertEqual(swap_behavior([value],[value])['pattern'],'ambiguous')

    def test_out_of_range_probabilities_are_rejected(self):
        for left,right in (([1.1],[.2]),([.4],[-.1])):
            with self.assertRaisesRegex(ValueError,'in \\[0, 1\\]'):
                swap_behavior(left,right)

    def test_unknown_or_ambiguous_declaration_is_not_confirmed(self):
        from superagent.task_semantics import declaration_consistency
        self.assertIsNone(declaration_consistency(dict(applicable=True,expected_pattern=None,pattern='flips')))
        self.assertIsNone(declaration_consistency(dict(applicable=True,expected_pattern='flips',pattern='ambiguous')))
        self.assertFalse(declaration_consistency(dict(applicable=True,expected_pattern='preserves',pattern='flips')))

    def test_complementary_predictions_mean_flips(self):
        forward = np.asarray([.9, .1, .75])
        report = swap_behavior(forward, 1 - forward)
        self.assertEqual(report['pattern'], 'flips')
        self.assertLessEqual(report['mean_complementarity_error'], SWAP_TOLERANCE)

    def test_identical_predictions_mean_preserves(self):
        forward = np.asarray([.9, .1, .75])
        self.assertEqual(swap_behavior(forward, forward)['pattern'], 'preserves')

    def test_inconsistent_predictions_are_neither(self):
        self.assertEqual(swap_behavior(np.asarray([.9]), np.asarray([.9]))['pattern'], 'preserves')
        self.assertEqual(swap_behavior(np.asarray([.9, .2]), np.asarray([.6, .1]))['pattern'], 'neither')

    def test_empty_or_malformed_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            swap_behavior([], [])
        with self.assertRaisesRegex(ValueError, 'equally sized'):
            swap_behavior([.5, .5], [.5])

    def test_a_symmetric_model_is_detected_by_the_probe(self):
        model = DummyClassifier(strategy='most_frequent').fit(np.zeros((4, 1)), np.asarray([1, 0, 1, 0]))
        report = probe_model_swap('pair_text', model, [('a', 'b'), ('c', 'd'), ('e', 'f')])
        self.assertTrue(report['applicable'])
        self.assertEqual(report['pattern'], 'preserves')
        self.assertEqual(report['rows'], 3)

    def test_swap_probe_does_not_apply_to_other_kinds(self):
        self.assertFalse(probe_model_swap('numeric_tabular', object(), [1, 2])['applicable'])


class ExecutionBoundaryTests(unittest.TestCase):
    def test_unknown_callable_is_not_safe_by_default(self):
        from superagent.models import Candidate
        from superagent.task_semantics import validate_candidate_semantics
        candidate=Candidate('unknown',{},lambda *args:None)
        with self.assertRaisesRegex(ValueError,'swap rule unknown'):
            validate_candidate_semantics('pair_text','symmetric',candidate)

    def test_resolved_recipe_is_checked_before_search_or_refit_training(self):
        from superagent import experiment_worker
        from superagent.recipe_models import materialize_recipe
        c=materialize_recipe(dict(kind='pair_text',analyzer='word',representation='latent',
                                 classifier='logistic',C=.3,dimensions=32))
        document={'schema_version':2,'kind':'pair_text','portfolio':'expanded','entries':{
            c.family:dict(spec=c.params,params=c.params,origin={})}}
        # A registered candidate bypasses the static menu; worker must still reject it.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            registry=root/'registry.json'
            import json
            registry.write_text(json.dumps(document),encoding='utf-8')
            for mode in ('search','refit'):
                payload=root/'payload.joblib';output=root/'output.joblib'
                joblib.dump(dict(kind='pair_text',pair_semantics='symmetric',mode=mode,
                    X_train=[('a','b'),('c','d')],y_train=np.asarray([0,1]),
                    X_search=[('a','b')],y_search=np.asarray([0])),payload)
                with patch.object(sys,'argv',['worker',str(payload),c.family,str(output),str(registry)]):
                    with self.assertRaisesRegex(ValueError,'cannot serve symmetric'):
                        experiment_worker.main()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
