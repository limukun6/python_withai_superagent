import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_b1_05_pair_pilot import pair_package
from superagent.ab_catalog import capabilities_for
from superagent.ab_symmetric import CHANNELS, ENCODER, UnorderedSentenceFeatures, candidates
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.models import Candidate, predict_with_model, probability_with_model
from superagent.task_semantics import candidate_swap_rule, validate_candidate_semantics


class TestInvariantPairMethods(unittest.TestCase):
    def test_frozen_features_encode_the_same_unique_text_batch_in_either_order(self):
        pairs = [('a', 'z'), ('x', 'a'), ('z', 'z')]
        batches = []
        def encode(texts):
            batches.append(texts)
            return np.asarray([[i, i + 0.5] for i in range(len(texts))]), len(texts)
        with patch('superagent.ab_sentence.encode', side_effect=encode):
            features = UnorderedSentenceFeatures().fit(pairs)
            np.testing.assert_array_equal(features.transform(pairs), features.transform([(r, l) for l, r in pairs]))
        self.assertEqual(batches, [['a', 'x', 'z'], ['a', 'x', 'z']])
        method = next(c for c in candidates(include_encoder=True) if c.family == ENCODER)
        self.assertEqual(candidate_swap_rule(method), 'preserves')

    def test_actual_representations_labels_and_probability_columns_survive_swap(self):
        train = [('red pear', 'pear red'), ('red pear', 'blue sky'), ('green leaf', 'green leaves'),
                 ('a frog', 'green leaf'), ('A', 'a'), ('', '!')]
        y = np.array(['equivalent', 'different', 'equivalent', 'different', 'equivalent', 'different'])
        query = [('red pear', 'pear RED'), ('red pear', 'blue sky'), ('', ''), ('x', '!'),
                 ('未知词 café', 'mot étranger'), ('the same', 'the same')]
        swapped = [(right, left) for left, right in query]
        for candidate in candidates():
            with self.subTest(candidate=candidate.family):
                model, forward = candidate.fit_predict(train, y, query)
                features = model.named_steps['unordered_pair']
                self.assertEqual((features.transform(query) != features.transform(swapped)).nnz, 0)
                np.testing.assert_array_equal(forward, predict_with_model('pair_text', model, swapped))
                classes, probabilities = probability_with_model('pair_text', model, query)
                other_classes, reversed_probabilities = probability_with_model('pair_text', model, swapped)
                np.testing.assert_array_equal(classes, other_classes)
                np.testing.assert_array_equal(probabilities, reversed_probabilities)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertEqual(set(classes), set(y))

    def test_resolution_uses_fitted_code_and_rejects_other_semantics(self):
        candidates_by_name = candidates()
        self.assertEqual({candidate.family for candidate in candidates_by_name}, set(CHANNELS))
        for candidate in candidates_by_name:
            self.assertEqual(candidate_swap_rule(candidate), 'preserves')
            for declared in (None, 'relation', 'preference'):
                with self.assertRaises(ValueError):
                    validate_candidate_semantics('pair_text', declared, candidate)
        forged = Candidate(list(CHANNELS)[0], {'swap_label_rule': 'preserves'}, lambda *rows: None)
        self.assertEqual(candidate_swap_rule(forged), 'unknown')
        with self.assertRaises(ValueError):
            validate_candidate_semantics('pair_text', 'symmetric', forged)

    def test_equivalence_manifest_and_relation_preference_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [('one', 'two'), ('a', 'b'), ('A', 'b')]
            labels = [0, 1, 0]
            train = root / 'train.zip'
            pair_package(train, rows, labels, 'symmetric', 'train')
            task = load_development_task(train)
            self.assertEqual(task.y_train.tolist(), labels)
            self.assertEqual(task.spec['input_schema']['pair_label_one'], 'equivalent')
            spec = TaskSpec.from_task(task).record()
            with patch.dict('os.environ', {'SUPERAGENT_MINILM_WEIGHTS': ''}):
                self.assertEqual(set(capabilities_for(spec, train_rows=3)), set(CHANNELS))
            spec['input_schema']['pair_label_one'] = 'left'
            self.assertFalse(capabilities_for(spec, train_rows=3))
            for meaning in ('relation', 'preference'):
                packet = root / (meaning + '.zip')
                pair_package(packet, rows, [0, 1, 0], meaning, 'train')
                typed = load_development_task(packet)
                self.assertEqual(typed.kind, 'pair_text' if meaning == 'preference' else 'text_tabular')
                catalog = capabilities_for(TaskSpec.from_task(typed).record(), train_rows=3)
                self.assertTrue(catalog)
                self.assertFalse(set(CHANNELS).intersection(catalog))
            wrong = root / 'different-semantics.zip'
            pair_package(wrong, rows, None, 'preference', 'test')
            with self.assertRaisesRegex(ValueError, 'Prediction schema differs'):
                load_prediction_features(wrong, 'pair_text', task.spec)
