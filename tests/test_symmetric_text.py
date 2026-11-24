import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np

from superagent.ab_catalog import capabilities_for
from superagent.models import Candidate, probability_with_model, predict_with_model
from superagent.symmetric_text import SymmetricTfidf, fit_symmetric_pair, candidates
from superagent.task_semantics import candidate_swap_rule, validate_candidate_semantics
from tests.test_ab_scheduler import recorded


ROWS = [('red cat sleeps', 'red cat sleeps'), ('blue dog runs', 'blue dog runs'),
        ('green bird sings', 'green bird sings'), ('white fish swims', 'white fish swims'),
        ('red cat sleeps', 'blue dog runs'), ('blue dog runs', 'green bird sings'),
        ('green bird sings', 'white fish swims'), ('white fish swims', 'red cat sleeps')]
LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])


class SymmetricTextTests(unittest.TestCase):
    def test_both_feature_routes_exactly_preserve_swaps_and_exclude_query_vocabulary(self):
        for representation in ('word', 'word_char'):
            transform = SymmetricTfidf(representation).fit(ROWS)
            query = [('heldoutonly moon', 'red cat sleeps'), ('', 'blue dog runs')]
            forward = transform.transform(query)
            reverse = transform.transform([(b, a) for a, b in query])
            self.assertEqual((forward - reverse).nnz, 0)
            self.assertNotIn('heldoutonly', transform.vectorizers_[0].vocabulary_)
            self.assertEqual(transform.transform([]).shape[0], 0)

    def test_probability_labels_same_under_swap_and_joblib_reload(self):
        # Exactly two supervised fits; labels are unchanged for the reversed training input.
        previous = None
        for rows in (ROWS, [(b, a) for a, b in ROWS]):
            model, prediction = fit_symmetric_pair(rows, LABELS, ROWS, representation='word_char')
            classes, proba = probability_with_model('pair_text', model, ROWS)
            reverse = [(b, a) for a, b in ROWS]
            self.assertTrue(np.array_equal(prediction, predict_with_model('pair_text', model, reverse)))
            np.testing.assert_array_equal(proba, model.predict_proba(reverse))
            if previous is not None:
                np.testing.assert_array_equal(proba, previous)
            previous = proba
            self.assertEqual(set(classes), {0, 1})
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.joblib'
                joblib.dump(model, path)
                np.testing.assert_array_equal(proba, joblib.load(path).predict_proba(ROWS))

    def test_metadata_cannot_impersonate_a_known_symmetric_executor(self):
        fake = Candidate('fake', dict(swap_label_rule='preserves', swap_augmentation=False), lambda *args: None)
        self.assertEqual(candidate_swap_rule(fake), 'unknown')
        with self.assertRaises(ValueError):
            validate_candidate_semantics('pair_text', 'symmetric', fake)
        for candidate in candidates():
            self.assertEqual(candidate_swap_rule(candidate), 'preserves')
            with self.assertRaises(ValueError):
                validate_candidate_semantics('pair_text', 'preference', candidate)

    def test_symmetric_and_preference_catalogues_are_separate_and_worker_resolves_all(self):
        from superagent.models import search_candidates
        from superagent.ab_run import registered_capabilities
        task = recorded()['task']
        task.update(kind='pair_text', input_schema=dict(pair_semantics='symmetric', pair_label_one='equivalent'))
        caps = capabilities_for(task, train_rows=200)
        native = [candidate.family for candidate in candidates()]
        self.assertEqual(len(native), 4)
        self.assertEqual(list(caps)[:4], native)
        self.assertEqual(len(registered_capabilities(task, 200, 'baseline')), 0)
        worker = {c.family: c for c in search_candidates('pair_text', 'all')}
        for name, entry in caps.items():
            self.assertEqual(entry['params'], worker[name].params)
            self.assertEqual(entry['swap_label_rule'], 'preserves')
            self.assertEqual((entry['classifier_fits'], entry['refit_classifier_fits']), (1, 1))
        task['input_schema'] = dict(pair_semantics='preference', pair_label_one='left')
        preference = capabilities_for(task, train_rows=200)
        self.assertTrue(preference)
        self.assertFalse(set(preference) & set(caps))
        task['input_schema'] = dict(pair_semantics='relation')
        self.assertFalse(capabilities_for(task, train_rows=200))
        task['input_schema'] = {}
        self.assertFalse(capabilities_for(task, train_rows=200))

    def test_bad_input_and_unknown_representation_fail_explicitly(self):
        with self.assertRaises(ValueError):
            SymmetricTfidf('unsupported').fit(ROWS)
        for bad in ([], ['not a pair'], [('one', 2)], [None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SymmetricTfidf().fit(bad)


if __name__ == '__main__':
    unittest.main()
