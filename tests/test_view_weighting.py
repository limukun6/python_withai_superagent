import unittest

import numpy as np
from scipy.sparse import csr_matrix

from superagent.view_weighting import scales


class FeatureBlockEvidence(unittest.TestCase):
    def test_supervised_signal_requires_agreement_across_groups(self):
        labels = np.tile([0, 1], 20)
        groups = np.repeat(np.arange(20), 2)
        signal = csr_matrix((2 * labels - 1).reshape(-1, 1))
        constant = csr_matrix(np.ones((len(labels), 1)))
        weights, evidence = scales([signal, constant], "aligned", labels, groups)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(evidence[0], evidence[1])
        inverted, _ = scales([signal, constant], "aligned", 1 - labels, groups)
        np.testing.assert_allclose(weights, inverted)
        with self.assertRaises(ValueError):
            scales([signal], "aligned", labels, groups[:-1])

    def test_no_informative_block_and_label_free_mode(self):
        groups = np.repeat(np.arange(4), 2)
        labels = np.tile([0, 1], 4)
        zero = csr_matrix((len(labels), 2))
        ones = csr_matrix(np.ones((len(labels), 1)))
        weights, _ = scales([ones, zero], "aligned", labels, groups)
        np.testing.assert_allclose(weights, [1, 1])
        first, _ = scales([ones, zero], "energy", labels, groups)
        other, _ = scales([ones, zero], "energy", 1 - labels, groups[::-1])
        np.testing.assert_allclose(first, other)
