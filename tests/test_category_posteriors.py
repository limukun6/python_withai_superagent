import unittest

import numpy as np

from superagent.category_posteriors import CategoryPosteriors
from superagent.evidence_agent import route_of
from superagent.method_choice import COOCCUR, PAIR_POSTERIOR, SINGLE_POSTERIOR, select_action


class CategoryPosteriorsTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.asarray([[f"a{i%4}", f"b{i%3}", f"c{i%7}"] for i in range(84)] * 2)
        self.labels = np.array([i%3 for i in range(len(self.rows))])

    def test_group_cannot_use_its_own_labels(self):
        members = np.all(self.rows == self.rows[0], axis=1)
        changed = self.labels.copy()
        changed[members] = (changed[members]+1)%3
        first = CategoryPosteriors(True).fit_transform(self.rows, self.labels)
        second = CategoryPosteriors(True).fit_transform(self.rows, changed)
        np.testing.assert_array_equal(first[members], second[members])
        self.assertGreater(np.max(np.abs(first[~members]-second[~members])), 0)

    def test_unknown_fields_have_train_prior_and_pairs_back_off(self):
        encoder = CategoryPosteriors(True)
        features = encoder.fit_transform(self.rows, self.labels)
        self.assertEqual(features.shape, (168, 12))
        unknown = np.asarray([["new", "unknown", "unseen"]])
        np.testing.assert_allclose(encoder.transform(unknown), np.full((1,12),1/3))
        x = encoder.transform(np.asarray([["new", "b0", "unseen"]])).reshape(6,2)
        for j, fields in enumerate(encoder.fields):
            if len(fields) == 2:
                np.testing.assert_allclose(x[j], (x[fields[0]]+x[fields[1]])/2, atol=1e-6)

    def test_second_trial_depends_on_feedback(self):
        evidence = dict(kind="categorical_tabular", semantic="unused", shared_errors=28, required_shared_errors=10,
                        short_latin_share=.9, low_support_shared_errors=15, primary_metric="accuracy", corrected=17, lost=6)
        feasible = {PAIR_POSTERIOR, SINGLE_POSTERIOR, COOCCUR}
        self.assertEqual(route_of(PAIR_POSTERIOR), "grouped_category_effects")
        self.assertEqual(select_action(evidence, feasible, "posteriors", [])[0], PAIR_POSTERIOR)
        self.assertEqual(select_action(evidence, feasible, "posteriors", [PAIR_POSTERIOR])[0], SINGLE_POSTERIOR)
        self.assertEqual(select_action(evidence | {"corrected":3}, feasible, "posteriors", [PAIR_POSTERIOR])[0], COOCCUR)

    def test_missing_levels_are_batch_invariant(self):
        rows = self.rows.astype(object)
        rows[::3, 0] = "?"
        encoder = CategoryPosteriors(True)
        encoder.fit_transform(rows, self.labels)
        expected = encoder.transform([["?", "b0", "c0"]])
        actual = encoder.transform([["NaN", "b0", "c0"], ["a very long new value", "b1", "c1"]])
        np.testing.assert_array_equal(actual[:1], expected)
