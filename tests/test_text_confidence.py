import unittest

import numpy as np

from superagent.text_confidence import regularization_evidence


class TextConfidenceTests(unittest.TestCase):
    def test_underconfidence_and_overconfidence_point_in_opposite_directions(self):
        probs = np.tile([.4, .6], (32, 1))
        groups = np.arange(32) // 4
        for label, expected in (("b", 1.2), ("a", .075)):
            evidence = regularization_evidence(probs, ["a", "b"], [label]*32, groups, .3)
            self.assertEqual(evidence["next_c"], expected)
            self.assertEqual(evidence["group_count"], 8)

    def test_indistinguishable_or_dependent_probabilities_skip_the_trial(self):
        probs = np.tile([.5, .5], (18, 1))
        for groups in (np.zeros(18), np.arange(18)):
            self.assertIsNone(regularization_evidence(probs, [0, 1], np.arange(18)%2, groups, .3)["next_c"])
