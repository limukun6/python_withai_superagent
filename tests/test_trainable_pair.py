from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from superagent import trainable_pair as learned
from superagent.models import search_candidates, predict_with_model, probability_with_model
from superagent.splits import overlap_audit, take
from superagent.data import DevelopmentTask, TrainingTask
from superagent.evidence_agent import route_of
from superagent.cli import build_parser
from superagent.benchmark import run_one


def fixture(n=100):
    pairs = [(f'bright warm signal item{i} sidea' if i % 2 else f'cold quiet noise item{i} sidea',
              f'cold quiet noise item{i} sideb' if i % 2 else f'bright warm signal item{i} sideb') for i in range(n)]
    return pairs, np.arange(n) % 2


class TrainablePairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def quick_neural(self, x, y, search):
        with patch.object(learned, 'MAX_EPOCHS', 3), patch.object(learned, 'PATIENCE', 2):
            return learned.fit_neural_pair(x, y, search)

    def test_neural_prediction_inputs_do_not_change_training_or_vocab(self):
        x, y = fixture()
        a, _ = self.quick_neural(x, y, [('privateheldouttoken', 'unknownword')])
        b, _ = self.quick_neural(x, y, [('entirelydifferentheldout', '')])
        self.assertEqual(a.words, b.words)
        self.assertNotIn('privateheldouttoken', a.words)
        self.assertEqual(a.training_metadata, b.training_metadata)
        for k in a.weights: np.testing.assert_array_equal(a.weights[k], b.weights[k])

    def test_neural_selector_fits_vocabulary_only_on_selector_training(self):
        x, y = fixture(); model, _ = self.quick_neural(x, y, x[:3])
        meta = model.training_metadata
        expected = learned.vocabulary(take(x, meta['selector_fit_indices']))
        self.assertEqual(expected, meta['selector_vocabulary'])
        self.assertEqual(meta['selector_overlap_audit']['search_rows_sharing_text'], 0)
        curve = meta['selector_curve']
        self.assertEqual(meta['selected_iterations'], min(curve, key=lambda r: (r['holdout_loss'], r['epoch']))['epoch'])
        self.assertEqual(len(meta['refit_curve']), meta['selected_iterations'])
        self.assertEqual(model.classifier_fits, 2)

    def test_neural_swap_empty_unknown_and_serialization(self):
        x, y = fixture(); model, _ = self.quick_neural(x, y, x[:3])
        pairs = [x[2], ('', ''), ('neverencountered', 'differentunseen')]
        before = deepcopy(model.words)
        p = model.predict_proba(pairs)[:, 1]
        q = model.predict_proba([(b, a) for a, b in pairs])[:, 1]
        np.testing.assert_allclose(p + q, 1., atol=1e-7)
        self.assertEqual(p[1], .5)
        self.assertEqual(model.words, before)
        file = self.root / 'neural.joblib'; joblib.dump(model, file)
        restored = joblib.load(file); self.assertIsNone(restored._network)
        np.testing.assert_array_equal(model.predict(pairs), predict_with_model('pair_text', restored, pairs))
        np.testing.assert_allclose(probability_with_model('pair_text', restored, pairs)[1][:, 1], p)

    def test_invalid_labels_dimensions_or_group_count_fail(self):
        x, y = fixture()
        for dimension in (True, 16, 0):
            with self.assertRaises(ValueError): learned.fit_neural_pair(x, y, x[:1], dimension)
        with self.assertRaises(ValueError): learned.fit_neural_pair(x, np.zeros(len(y)), x[:1])
        with self.assertRaises(ValueError): learned.fit_neural_pair([('a', 'b')] * 10, [0, 1] * 5, x[:1])

    def test_complementarity_gate_does_not_report_oracle_as_achievement(self):
        y = np.array([0, 1] * 20)
        perfect = np.where(y, 2., -2.)
        same = learned.complementary_gate(np.column_stack([perfect] * 3), y)
        self.assertFalse(same['requested']); self.assertEqual(same['disagreement_fraction'], 0.)
        a, b = perfect.copy(), perfect.copy(); a[:8] *= -1; b[8:16] *= -1
        different = learned.complementary_gate(np.column_stack([a, b, a]), y)
        self.assertTrue(different['requested']); self.assertFalse(different['oracle_is_achievable_score'])
        self.assertAlmostEqual(different['oracle_accuracy'], 1.)

    def test_oof_models_are_disjoint_and_cover_every_row_once(self):
        x, y = fixture()
        with patch.object(learned, 'MAX_EPOCHS', 2):
            model, _ = learned.fit_learned_stack(x, y, [('heldoutprivatetoken', 'newword')])
        meta = model.training_metadata
        self.assertEqual(meta['coverage_counts'], [1] * len(y))
        margins = np.array(meta['oof_margins'])
        for fold in model.audit_folds:
            fit, held = fold['fit'], fold['heldout']
            self.assertEqual(overlap_audit(TrainingTask('pair_text', x, y), fit, held)['search_rows_sharing_text'], 0)
            for column, component in enumerate(fold['models']):
                np.testing.assert_allclose(learned.component_margin(component, take(x, held)), margins[held, column])
                if isinstance(component, learned.NeuralPairModel):
                    self.assertEqual(component.words, learned.vocabulary(take(x, fit)))
                    self.assertNotIn('heldoutprivatetoken', component.words)
        expected = sum(sum(getattr(m, 'classifier_fits', 1) for m in f['models']) for f in model.audit_folds)
        expected += sum(getattr(m, 'classifier_fits', 1) for m in model.models) + int(model.head is not None)
        self.assertEqual(model.classifier_fits, expected)

    def test_failed_complementarity_gate_retains_best_component_without_meta_fit(self):
        x, y = fixture()
        original = learned.complementary_gate
        def gate(*args): return {**original(*args), 'requested': False}
        with patch.object(learned, 'MAX_EPOCHS', 2), patch.object(learned, 'complementary_gate', gate):
            model, _ = learned.fit_learned_stack(x, y, x[:2])
        self.assertIsNone(model.head)
        self.assertEqual(model.component_names, [model.training_metadata['complementarity_gate']['best_component']])
        self.assertIn(model.classifier_fits, (13, 14))

    def test_fusion_head_uses_oof_margins_and_preserves_swap_after_reload(self):
        x, y = fixture(); original = learned.complementary_gate
        def gate(*args): return {**original(*args), 'requested': True}
        with patch.object(learned, 'MAX_EPOCHS', 2), patch.object(learned, 'complementary_gate', gate):
            model, _ = learned.fit_learned_stack(x, y, x[:3])
        margins = np.array(model.training_metadata['oof_margins'])
        self.assertEqual(model.classifier_fits, 17)
        np.testing.assert_allclose(model.head.named_steps['scale'].scale_, np.std(np.vstack([margins, -margins]), axis=0))
        self.assertFalse(model.head.named_steps['classifier'].fit_intercept)
        file = self.root / 'stack.joblib'; joblib.dump(model, file); model = joblib.load(file)
        p = model.predict_proba(x[:8])[:, 1]; q = model.predict_proba([(b, a) for a, b in x[:8]])[:, 1]
        np.testing.assert_allclose(p + q, 1., atol=1e-6)

    def test_optional_catalog_routes_and_cli_are_usable(self):
        base = search_candidates('pair_text', 'expanded'); new = search_candidates('pair_text', 'learned')
        self.assertEqual([r.family for r in base], [r.family for r in new[:22]])
        self.assertEqual(len(new), 25)
        self.assertEqual([route_of(r.family) for r in new[22:]], ['neural_pair', 'neural_pair', 'learned_stack'])
        self.assertEqual(build_parser().parse_args(['run', 'task.zip', '--run-dir', 'run', '--portfolio', 'learned']).portfolio, 'learned')

    def test_neural_worker_confirmation_and_full_refit_count_every_selector(self):
        x, y = fixture(130)
        task = DevelopmentTask('pair_text', 'synthetic', x[:100], y[:100], x[100:], y[100:], {'metric': 'accuracy'})
        result = run_one(task, 'local_interleaved_routes', 707, self.root / 'pipeline', None,
            max_trials=1, portfolio='learned', catalog_ids=['neural_pair_dim32'],
            confirmation_repeats=3, confirmation_top_k=1, confirmation_grouping='shared_text',
            search_grouping='shared_text', refit_full_train=True, time_limit=180, trial_timeout=60)
        self.assertTrue(result['valid_policy_run']); self.assertTrue(result['refit_succeeded'])
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'], 8)
        self.assertEqual(result['confirmation']['classifier_fits_completed'], 4)
        self.assertEqual(result['refit_classifier_fits_completed'], 2)


if __name__ == '__main__': unittest.main()
