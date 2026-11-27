import unittest

import numpy as np

from superagent.group_exposure import errors, proposals, sizes


class TestGroupExposure(unittest.TestCase):
    def test_connected_sizes_and_error_direction(self):
        rows = [("a", "b"), ("z", "y"), ("B", "c")]
        np.testing.assert_array_equal(sizes(rows), [2,1,2])
        counts = np.repeat([1,2], 40)
        y = np.ones(80)
        evidence = errors(counts, y, np.r_[np.ones(40), np.zeros(40)])
        self.assertTrue(evidence["distinguishes"])
        name = "gte_paired_linear"
        row = dict(candidate=name, split_id="primary", status="completed", search_score=.5, connected_group_errors=evidence)
        specs = proposals([row], {name})
        self.assertEqual({s["power"] for s in specs.values()}, {-.5,.5})
        self.assertEqual(proposals([dict(row, split_id="confirmation_1")], {name}), {})
        self.assertEqual(proposals([row], set()), {})

    def test_zero_gap_and_short_slices_do_not_nominate(self):
        self.assertIsNone(errors(np.ones(10), np.ones(10), np.ones(10)))
        counts = np.repeat([1,2],40)
        self.assertFalse(errors(counts,np.ones(80),np.ones(80))["distinguishes"])
