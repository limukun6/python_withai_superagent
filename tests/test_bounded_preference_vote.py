import unittest
from unittest.mock import patch

import numpy as np

from superagent.ab_complement import PreferenceProbabilityVote
from superagent.bounded_vote import BoundedPreferenceVote


class BoundedVoteTests(unittest.TestCase):
    def votes(self, scores, *, dtype=np.float64, perturb=False):
        scores = np.asarray(scores, dtype=dtype)
        members = list(range(3))
        rows = [(f"a{i}", f"z{i}") for i in range(len(scores))]
        def probabilities(model, pairs):
            values = []
            for a, b in pairs:
                i = int(a[1:])
                p = scores[i, model]
                if perturb and model == 2 and len(pairs) < len(rows):
                    p -= .00002
                values.append(p if a < b else 1 - p)
            p = np.array(values, dtype=dtype)
            return np.array([0, 1]), np.column_stack([1 - p, p])
        with patch("superagent.bounded_vote.canonical_probabilities", side_effect=probabilities), \
                patch("superagent.ab_complement.canonical_probabilities", side_effect=probabilities):
            full = PreferenceProbabilityVote("pair_text", members, np.array([0, 1]))
            fast = BoundedPreferenceVote(members)
            output, work = fast.predict_with_counts(rows)
            np.testing.assert_array_equal(output, full.predict(rows))
            np.testing.assert_array_equal(fast.predict_proba(rows), full.predict_proba(rows))
            flipped = [(b, a) for a, b in rows]
            np.testing.assert_array_equal(fast.predict(flipped), full.predict(flipped))
        return work

    def test_extremes_skip_third_around_exact_half_and_one_and_a_half(self):
        work = self.votes([[.10, .10, .99], [.91, .80, 0], [.25, .25, .9], [.75, .75, .1],
                           [.26, .25, .97], [.75, .74, .03]])
        self.assertEqual((work["certified"], work["third_rows"], work["full_batch_rechecks"]), (2, 4, 0))

    def test_near_tie_recomputes_the_original_third_batch(self):
        work = self.votes([[.1, .1, .4], [.6, .5, .40001]], perturb=True)
        self.assertEqual((work["certified"], work["third_rows"], work["full_batch_rechecks"]), (1, 3, 1))
        ties = self.votes([[.25, .25, 1], [.75, .75, 0]])
        self.assertEqual((ties["certified"], ties["third_rows"]), (0, 2))

    def test_random_probabilities_match_all_outputs_and_low_precision_keeps_full_work(self):
        p = np.random.default_rng(3012).random((1200, 3))
        self.assertGreater(self.votes(p)["certified"], 0)
        low = self.votes([[.1, .1, 1], [.8, .9, 0]], dtype=np.float32)
        self.assertEqual((low["certified"], low["third_rows"]), (0, 2))

    def test_empty_and_all_decided_batches(self):
        model = BoundedPreferenceVote([0, 1, 2])
        self.assertEqual(model.predict([]).size, 0)
        self.assertEqual(model.predict_proba([]).shape, (0, 2))
        all_decided = self.votes([[.1, .1, 1], [.8, .9, 0]])
        self.assertEqual(all_decided["third_rows"], 0)
