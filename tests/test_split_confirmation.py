import unittest
import numpy as np

from tools.split_confirmation import choose_separately, feedback_halves


class SplitConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat([f"group_{i}" for i in range(64)],2)
        self.y = np.tile([0,1],64)
        self.ties = self.y.astype(bool)
        self.base = np.full(len(self.y),.7)
        self.expert = .2+.6*self.y

    def select(self, candidate):
        return choose_separately({"baseline":self.base,"candidate":self.expert},self.y,self.ties,
                                 {"baseline":self.base,"candidate":candidate},self.y,self.ties,self.groups)

    def test_two_roles_do_not_share_groups_and_do_not_depend_on_row_order(self):
        a,b = feedback_halves(self.groups)
        self.assertFalse(set(self.groups[a]) & set(self.groups[b]))
        np.testing.assert_array_equal(a[::-1],feedback_halves(self.groups[::-1])[0])

    def test_ratification_cannot_change_the_nomination_after_seeing_its_labels(self):
        result = self.select(self.expert)
        self.assertEqual((result["nominee"],result["selected"]),("candidate","candidate"))
        _,b = feedback_halves(self.groups)
        reversed_on_gate = self.expert.copy()
        reversed_on_gate[b] = 1-reversed_on_gate[b]
        bad = self.select(reversed_on_gate)
        self.assertEqual(result["nominations"],bad["nominations"])
        self.assertEqual((bad["nominee"],bad["selected"]),("candidate","baseline"))

    def test_no_learning_advantage_or_too_few_groups_keeps_the_incumbent(self):
        result = self.select(self.base)
        self.assertEqual(result["selected"],"baseline")
        self.groups = np.repeat("same",len(self.y))
        self.assertIsNone(self.select(self.expert)["ratification"])
