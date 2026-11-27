import unittest

from superagent.preference_order import inspect_order


class TestPreferenceOrder(unittest.TestCase):
    def test_chain_excludes_direct_answer_and_conflicts(self):
        evidence, pairs = inspect_order([("a","b"),("b","c")], [1,1])
        self.assertEqual(pairs,[("a","c")])
        self.assertEqual(evidence["checkable_pairs"],0)
        evidence, pairs = inspect_order([("a","b"),("b","c"),("a","c")], [1,1,1])
        self.assertEqual((evidence["checkable_pairs"], evidence["correct_pairs"]), (1,1))
        self.assertEqual(pairs, [])
        evidence, pairs = inspect_order([("a","b"),("b","c"),("a","c"),("c","a")], [1,1,1,1])
        self.assertEqual((evidence["conflicting_observations"],evidence["checkable_pairs"]), (1,0))
        self.assertEqual(pairs, [])

    def test_cycle_exposes_wrong_weak_labels(self):
        evidence, pairs = inspect_order([("a","b"),("b","c"),("c","a")], [1,1,1])
        self.assertEqual((evidence["correct_pairs"],evidence["checkable_pairs"]), (0,3))
        self.assertFalse(evidence["eligible"])
