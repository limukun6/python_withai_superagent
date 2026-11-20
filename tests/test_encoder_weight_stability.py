import unittest
import numpy as np

from tools.encoder_weight_stability import CONTROLS, accuracy_weight, global_limit


class StabilityTests(unittest.TestCase):
    def test_inner_accuracy_ties_keep_the_loss_weight_and_duplicate_visits_count_once(self):
        labels = np.array([1, 0])
        scores = np.array([[.8,.3],[.7,.2]])
        single = accuracy_weight([scores], [np.array([0,1])], labels, .5)
        repeated = accuracy_weight([scores,scores[:1],scores[:1]], [np.array([0,1]),[0],[0]], labels, .5)
        self.assertEqual(single, .5)
        self.assertEqual(repeated, single)

    def test_one_weight_cannot_reconcile_opposite_partition_preferences(self):
        blocks = []
        for seed,label in ((1,0),(2,1)):
            for _ in range(5):
                blocks.append(dict(seed=seed,y=np.array([label]),p=np.array([.9]),q=np.array([.1]),
                                   control_counts=dict.fromkeys(CONTROLS,0)))
        result = global_limit(blocks)
        self.assertFalse(result['any_fixed_weight_qualifies_everywhere'])
        self.assertEqual(result['upper_best_worst_margin']['worst_margin_rows'], 0)
        for b in blocks:
            b['y'] = np.array([1])
        self.assertTrue(global_limit(blocks)['any_fixed_weight_qualifies_everywhere'])
