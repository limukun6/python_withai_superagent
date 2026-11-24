import unittest

import numpy as np

from scripts.prepare_conditional_handoff import partition
from tools.preference_handoff import POLICIES, fit_policy, inputs, predict, predictions, select


class PreferenceHandoffTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(41)
        count = 400
        base = rng.uniform(.51, .9, count)
        third = rng.uniform(.03, .49, count)
        p = np.column_stack([base, np.minimum(.99, base + .025), third])
        pairs = [("one small revision", "another revision")] * count
        # Only the challenger confidence is informative on this synthetic example.
        y = np.where(third < .23, 0, 1)
        return p, pairs, y

    def test_switches_are_learned_and_swap_invariant(self):
        p, pairs, y = self.sample()
        learn = inputs(p[:220], pairs[:220])
        other = inputs(p[220:], pairs[220:])
        swapped = inputs(1-p[220:], [tuple(reversed(r)) for r in pairs[220:]])
        policies = {n: fit_policy(n, learn, y[:220]) for n in POLICIES}
        decision = select(policies, learn, y[:220], other, y[220:])
        self.assertNotEqual(decision["chosen"], "baseline")
        for policy in policies.values():
            np.testing.assert_array_equal(predict(policy, other), 1-predict(policy, swapped))
        actual = predictions(other, policies, decision)["agent"]
        self.assertGreater((actual == y[220:]).sum(), (other["base"] == y[220:]).sum())
        self.assertGreater(np.count_nonzero(actual != other["base"]), 0)
        self.assertGreater(np.count_nonzero(actual == other["base"]), 0)

    def test_negative_confirmation_keeps_control_and_agreements_never_switch(self):
        p, pairs, y = self.sample()
        d = inputs(p, pairs)
        policies = {n: fit_policy(n,d,y) for n in POLICIES}
        choice = select(policies,d,y,d,d["base"])
        self.assertEqual(choice["chosen"], "baseline")
        agree = inputs(np.full((len(pairs),3),.9), pairs)
        for policy in policies.values():
            np.testing.assert_array_equal(predict(policy, agree), agree["base"])
        self.assertEqual(fit_policy("hard", agree, y)["supervised_fits"], 0)

    def test_outer_folds_cover_each_new_group_once(self):
        groups = np.asarray([f"g{i//2}" for i in range(200)])
        reserved = np.arange(20)
        outer = []
        for role in partition(groups, reserved):
            group_sets = [set(groups[ids]) for ids in role.values()]
            self.assertEqual(sum(map(len,group_sets)), len(set.union(*group_sets)))
            self.assertTrue(set(reserved) <= set(role["base"]))
            outer.extend(role["outer"])
        self.assertEqual(sorted(outer), list(range(20,200)))
