"""Regression coverage for the optional held-out random seed decision."""
import unittest

import numpy as np

from superagent.metrics import MetricSpec
from superagent.seed_replication import evidence
from superagent.training_randomness import selected_seed, training_seed


class SeedReplicationTests(unittest.TestCase):
    def test_worker_override_is_scoped_and_default_remains_available(self):
        self.assertEqual(selected_seed(42), 42)
        with training_seed(1729):
            self.assertEqual(selected_seed(42), 1729)
            with training_seed(None):
                self.assertEqual(selected_seed(42), 42)
        self.assertIsNone(selected_seed())
        with self.assertRaises(ValueError):
            with training_seed(True):
                pass

    def test_identical_predictions_do_not_trigger_more_training(self):
        y = np.tile([0, 1], 120)
        p = np.column_stack((.7 - .4*y, .3 + .4*y))
        result = evidence(MetricSpec("accuracy", (0,1)), y, p, p, np.arange(len(y)))
        self.assertFalse(result["accept"])
        self.assertEqual(result["rms_probability_change"], 0)

    def test_declared_accuracy_and_probability_quality_both_matter(self):
        y = np.tile([0, 1], 160)
        p = np.column_stack((.65 - .3*y, .35 + .3*y))
        better = np.column_stack((.75 - .5*y, .25 + .5*y))
        for proposal, expected in ((better, True), (p[:, ::-1], False)):
            result = evidence(MetricSpec("accuracy", (0,1)), y, p, proposal, np.arange(len(y)))
            self.assertEqual(result["accept"], expected)
