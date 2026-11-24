import unittest

import numpy as np

from tools.pilot_challenger import approve, nominate, panel_scores, pilot_roles


class PilotChallengerTests(unittest.TestCase):
    def test_disjoint_groups_invariant_to_order(self):
        groups = np.repeat([f"g{i}" for i in range(90)], 2)
        y = np.tile([0, 1], 90)
        roles = pilot_roles(groups, y)
        perm = np.random.default_rng(82).permutation(len(y))
        shuffled = pilot_roles(groups[perm], y[perm])
        positions = []
        for role, ids in roles.items():
            self.assertEqual(set(groups[ids]), set(groups[perm][shuffled[role]]))
            positions.extend(set(groups[ids]))
        self.assertEqual(len(positions), len(set(groups)))
        self.assertEqual(set(positions), set(groups))
        self.assertIsNone(pilot_roles(np.array(["only"] * 20), np.tile([0, 1], 10)))

    def test_feedback_changes_nominee_and_confirmation_can_veto(self):
        n = 120
        y = np.tile([0, 1], n // 2)
        a = .15 + .7 * y
        rows = [(str(i), str(i+1)) for i in range(n)]
        g = np.array([str(i//2) for i in range(n)])
        values = np.column_stack((np.full(n, .52), np.full(n, .52), a, 1-a))
        names = ["anchor1", "anchor2", "x", "y"]
        learn = panel_scores(names, values, rows, y, g)
        flipped = panel_scores(names, values, rows, 1-y, g)
        for strategy in ("pilot-accuracy", "pilot-brier"):
            self.assertEqual(nominate(learn, strategy), "x")
            self.assertEqual(nominate(flipped, strategy), "y")
            self.assertEqual(approve("x", learn, learn["x"], n)["approved"], "x")
            self.assertIsNone(approve("x", learn, flipped["x"], n)["approved"])
            flat = panel_scores(names, np.full_like(values, .5), rows, y, g)
            self.assertIsNone(nominate(flat, strategy))
            self.assertIsNone(approve(None, flat, None, n)["approved"])

    def test_names_only_resolve_exact_ties(self):
        row = dict(net=2, brier_gain=.01, brier_lower=.005)
        for strategy in ("pilot-accuracy", "pilot-brier"):
            self.assertEqual(nominate({"z": row, "a": row}, strategy), "a")
            self.assertEqual(nominate({"a": row, "z": row}, strategy), "a")
        self.assertIsNone(approve("z", {"z": row}, {"net": 1, "brier_gain": .2}, 200)["approved"])
