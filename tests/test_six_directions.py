from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np

from superagent.error_slices import ErrorSlices
from superagent.contracts import DecisionState
from superagent.decision_architecture import ArchitecturePolicy, wire_state
from superagent.events import RunLogger
from superagent.recipe_models import RecipeTextModel, materialize_recipe, semantic_candidates
from superagent.recipe_registry import RecipeRegistry, resolve_recipe_registry
from superagent.resource_allocation import nested_subsample, promote, candidate_pool, run_allocated
from superagent.models import predict_with_model, search_candidates
from superagent.data import DevelopmentTask
from superagent.benchmark import run_one
from test_decision_architecture import state_fixture, proposal, review, ScriptClient


class SixDirectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_error_counts_thresholds_and_vocabulary_use_training_only(self):
        model = ErrorSlices('pair_text', [('common news', 'ordinary news')] * 40)
        before = deepcopy(model.__dict__)
        x = [('unseen confidentialtoken', 'common news')] * 20 + [('common news', 'ordinary news')] * 20
        y = np.array([0, 1] * 20); pred = np.array([1, 0] * 10 + [0, 1] * 10)
        result = model.describe(x, y, pred)
        self.assertEqual(model.__dict__, before)
        high = next(r for r in result['slices'] if r['slice_id'] == 'unseen_token_fraction_high')
        self.assertEqual((high['rows'], high['correct']), (20, 0))
        self.assertEqual(high['threshold_from_training'], 0.)
        self.assertNotIn('confidentialtoken', json.dumps(result))
        self.assertFalse(result['individual_inputs_or_labels_exposed'])

    def test_feedback_projection_exposes_aggregates_only_when_visible(self):
        original = state_fixture().record()
        history = original['history']
        history[0]['error_slices'] = {'status': 'completed', 'slices': [{'slice_id': 'text_length_high', 'rows': 30, 'accuracy': .5}]}
        visible = DecisionState.build(original['profile'], history, original['available_candidates'], 3, True)
        blind = DecisionState.build(original['profile'], history, original['available_candidates'], 3, False)
        self.assertIn('latest_error_slices', wire_state(visible.record()))
        self.assertNotIn('error_slices', json.dumps(blind.record()))

    def policy(self, state, answers):
        logger = RunLogger(self.root); logger.initialize_usage_ledger()
        client = ScriptClient(answers)
        policy = ArchitecturePolicy(client, logger, 'selective_review')
        chosen = policy.propose(state)
        return policy, client, chosen

    def test_selective_same_route_skips_without_extra_call(self):
        policy, client, chosen = self.policy(state_fixture(), [proposal()])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(chosen.candidate, 'word_C1.0')
        self.assertEqual(policy.records[0]['status'], 'selective_review_skipped')
        self.assertTrue(policy.summary()['all_requested_decisions_valid'])

    def test_new_route_final_decision_and_warning_trigger_review(self):
        for cause in ('route', 'final', 'warning'):
            state = state_fixture().record(); answer = proposal()
            if cause == 'route': answer = proposal('char_C1.0')
            if cause == 'final': state['trial_budget']['remaining'] = 1
            if cause == 'warning': answer['reason'] = 'Test stronger regularization by raising C.'
            policy, client, chosen = self.policy(DecisionState(state), [answer, review()])
            self.assertEqual(len(client.calls), 2, cause)
            self.assertTrue(policy.records[0]['review_gate']['requested'])
            self.assertEqual(chosen.candidate, answer['candidate'])
            self.assertEqual(policy.records[0]['status'], 'review_keep')

    def test_selective_failed_review_retains_original_and_is_not_pure(self):
        policy, client, chosen = self.policy(state_fixture(), [proposal('char_C1.0'), ValueError('invalid review')])
        self.assertEqual(chosen.candidate, 'char_C1.0')
        self.assertFalse(policy.summary()['all_requested_decisions_valid'])
        with self.assertRaises(RuntimeError): policy.propose(state_fixture())
        self.assertEqual(len(client.calls), 2)

    def test_latent_fit_predict_and_serialization_do_not_fit_on_evaluation(self):
        x = [('bright sunny day ' + str(i), 'cold rainy night ' + str(i)) for i in range(20)]
        x += [(b, a) for a, b in x]; y = np.array([1] * 20 + [0] * 20)
        for representation in ('latent', 'hybrid', 'lexical'):
            spec = {'kind': 'pair_text', 'analyzer': 'word', 'representation': representation,
                    'classifier': 'logistic', 'C': .3, 'dimensions': 32}
            model = RecipeTextModel(spec).fit(x, y)
            vocabulary = dict(model.vectorizer.vocabulary_)
            values = [('newholdouttoken sunny', 'cold night')]
            expected = predict_with_model('pair_text', model, values)
            path = self.root / (representation + '.joblib'); joblib.dump(model, path)
            np.testing.assert_array_equal(expected, predict_with_model('pair_text', joblib.load(path), values))
            self.assertEqual(vocabulary, model.vectorizer.vocabulary_)
            self.assertNotIn('newholdouttoken', vocabulary)

    def test_recipe_bounds_and_registry_tamper_rejected(self):
        spec = semantic_candidates('pair_text')[0].params
        for name, value in [('C', True), ('C', 99), ('dimensions', 999), ('classifier', 'eval'), ('kind', 'task2')]:
            changed = {**spec, name: value}
            with self.assertRaises(ValueError): materialize_recipe(changed)
        registry = RecipeRegistry('pair_text', 'expanded', self.root / 'registry.json')
        catalog = search_candidates('pair_text', 'expanded')
        history = [{'trial': i + 1, 'candidate': c.family, 'params': c.params, 'status': 'completed',
                    'search_accuracy': .6 + .01 * i, 'generalization_gap': .2} for i, c in enumerate(catalog[:2])]
        added = registry.propose(history)
        self.assertEqual(len(added), 2)
        for row in added:
            self.assertEqual(row['origin']['after_trial'], 2)
            self.assertEqual(resolve_recipe_registry('pair_text', row['id'], registry.document).params, row['params'])
        bad = deepcopy(registry.document); bad['entries'][added[0]['id']]['params']['C'] = 9
        with self.assertRaises(ValueError): resolve_recipe_registry('pair_text', added[0]['id'], bad)

    def test_nested_resource_samples_and_protected_alternative(self):
        y = np.repeat([0, 1, 2], 40)
        a = nested_subsample(y, .25, 7); b = nested_subsample(y, .5, 7)
        self.assertTrue(set(a) <= set(b)); self.assertEqual(set(y[a]), {0, 1, 2})
        rows = [{'candidate': c, 'status': 'completed', 'search_accuracy': value}
                for c, value in [('word_C0.1', .9), ('word_C0.3', .8), ('char_C0.1', .5), ('char_C0.3', .4)]]
        ids, reason = promote(rows)
        self.assertEqual(ids, ['word_C0.1', 'word_C0.3', 'char_C0.1'])
        self.assertIn('protected', reason)
        self.assertEqual(candidate_pool('pair_text', 'expanded', 6, 3), candidate_pool('pair_text', 'expanded', 6, 3))

    def test_composition_runs_through_worker_confirmation_and_refit(self):
        x = ['bright sun warm ' + str(i) if i % 2 else 'cold rain night ' + str(i) for i in range(90)]
        y = np.arange(90) % 2
        task = DevelopmentTask('single_text', 'synthetic', x[:70], y[:70], x[70:], y[70:], {})
        result = run_one(task, 'local_interleaved_routes', 707, self.root / 'composition', None,
                         max_trials=6, anchor_first=True, portfolio='expanded', refinement='compose',
                         error_feedback=True, confirmation_repeats=3, refit_full_train=True,
                         search_grouping='shared_text', confirmation_grouping='shared_text')
        self.assertTrue(result['valid_policy_run'])
        self.assertTrue(result['refit_succeeded'])
        self.assertGreater(result['refinement']['generated_candidates'], 0)
        rows = json.loads((self.root / 'composition/trials.json').read_text())
        self.assertTrue(any(r['candidate'].startswith('recipe_') for r in rows))

    def test_resource_pipeline_counts_screening_and_full_refit(self):
        rng = np.random.default_rng(38); x = rng.normal(size=(120, 4)); y = (x[:, 0] > 0).astype(int)
        task = DevelopmentTask('numeric_tabular', 'synthetic', x[:90], y[:90], x[90:], y[90:], {})
        result = run_allocated(task, 'local_interleaved_routes', 707, self.root / 'resources', None,
                              'successive', max_trials=4, portfolio='expanded', time_limit=120,
                              trial_timeout=20, anchor_first=True, confirmation_repeats=3,
                              confirmation_top_k=3, refit_full_train=True, refinement='off')
        self.assertTrue(result['refit_succeeded'])
        allocation = result['resource_allocation']
        self.assertEqual(allocation['screening_classifier_fits'], 7)
        self.assertEqual(result['trials_attempted'], 2)
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'], 14)
        self.assertTrue(result['confirmation']['valid_confirmation'])


if __name__ == '__main__':
    unittest.main()
