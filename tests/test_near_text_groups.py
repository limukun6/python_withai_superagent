import itertools
import random
import unittest

import numpy as np

from superagent.data import TrainingTask
from superagent.near_text import matching_indices, word_pairs
from superagent.splits import sample_groups


class NearTextGroupingTests(unittest.TestCase):
    def test_prefix_index_has_exact_jaccard_recall(self):
        rng = random.Random(41101)
        sets = [set(range(8)), set(range(10)), set(range(11)), set()]
        for _ in range(180):
            features = set(rng.sample(range(30), rng.randint(5, 26)))
            sets.extend([features, features | {34}, features - {0, 1, 2}])
        wanted = {(a, b) for a, b in itertools.combinations(range(len(sets)), 2)
                  if sets[a] and sets[b] and 5 * len(sets[a] & sets[b]) >= 4 * len(sets[a] | sets[b])}
        self.assertEqual(set(matching_indices(sets)), wanted)
        self.assertIn((0, 1), wanted)
        self.assertNotIn((0, 2), wanted)

    def test_transitive_title_groups_survive_row_and_side_permutation(self):
        root = "Ａ b c d e f g h i j k l"
        pairs = [(root, "left"), (root + " today", "shared short"), ("shared short", "other"),
                 ("short", "unique"), ("different", "isolated")]
        task = TrainingTask("pair_text", pairs, np.array([0, 1, 0, 1, 1]))
        exact = sample_groups(task, "shared_text")
        strict = sample_groups(task, "near_text")
        self.assertEqual((len(set(exact)), len(set(strict))), (4, 3))
        self.assertEqual(len(set(strict[:3])), 1)
        swapped = [(b, a) for a, b in reversed(pairs)]
        reverse = TrainingTask("pair_text", swapped, task.y_train[::-1])
        np.testing.assert_array_equal(sample_groups(reverse, "near_text")[::-1], strict)
        self.assertFalse(word_pairs("one two three four five"))
        self.assertEqual(word_pairs("Ａ, b c d e F"), word_pairs("a b c d e f"))

    def test_non_pair_tasks_keep_their_exact_feature_groups(self):
        numeric = TrainingTask("numeric_tabular", np.array([[0, 2], [0, 2], [4, 3]]), np.array([0, 1, 1]))
        np.testing.assert_array_equal(sample_groups(numeric, "near_text"), sample_groups(numeric, "exact_rows"))
