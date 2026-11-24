import unittest
from unittest.mock import patch

import numpy as np

from tools.consensus_veto import ConsensusPreferenceVote, adjusted, confirm, propose


class ConsensusVetoTests(unittest.TestCase):
    def setUp(self):
        self.p = np.vstack([np.tile([.56, .54, .02], (24, 1)), np.tile([.22, .94, .70], (24, 1))])
        self.ties = np.ones(48, dtype=bool)
        self.groups = np.arange(48)
        self.labels = np.ones(48, dtype=int)

    def test_positive_independent_evidence_retains_only_the_pair_on_agreement(self):
        nomination, _ = propose(self.p, self.ties, self.labels, self.groups)
        self.assertEqual(nomination, "agree")
        decision, check = confirm(nomination, self.p, self.ties, self.labels, self.groups)
        self.assertEqual(decision, nomination)
        self.assertEqual(check["net"], 24)
        out = adjusted(self.p, self.ties, decision)
        np.testing.assert_allclose(out[:24], self.p[:24, :2].mean(1))
        np.testing.assert_allclose(out[24:], self.p[24:].mean(1))

    def test_a_negative_check_or_too_few_switches_keeps_the_global_vote(self):
        self.assertIsNone(confirm("agree", self.p, self.ties, 1 - self.labels, self.groups)[0])
        self.assertIsNone(propose(self.p[:9], self.ties[:9], self.labels[:9], self.groups[:9])[0])
        self.assertIsNone(propose(self.p, self.ties, self.labels, np.zeros(48, dtype=int))[0])
        np.testing.assert_allclose(adjusted(self.p, self.ties, None), self.p.mean(1))

    def test_model_predicts_swapped_rows_and_matches_the_selected_probability(self):
        def component(probability, rows):
            positive = np.array([.5 if a == b else probability if a < b else 1 - probability for a, b in rows])
            return np.array([0, 1]), np.column_stack([1 - positive, positive])
        with patch("tools.consensus_veto.canonical_probabilities", side_effect=component):
            model = ConsensusPreferenceVote([.56, .54, .02], "agree")
            rows = [("a", "b"), ("b", "a"), ("a", "a")]
            np.testing.assert_allclose(model.predict_proba(rows)[:, 1], [.55, .45, .5])
            np.testing.assert_array_equal(model.predict(rows), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
