import itertools
import random
import unittest

from scripts.audit_pilot_title_overlap import compute, matching_pairs, shingles
from scripts.study_external_preference import hashed, make_pairs


class OverlapTests(unittest.TestCase):
    def test_index_matches_exact_jaccard_including_the_boundary(self):
        rng = random.Random(1229)
        sets = [set(range(8)), set(range(10)), set(range(11)), set()]
        sets.extend(set(rng.sample(range(18), rng.randint(5, 14))) for _ in range(60))
        expected = {(a, b) for a, b in itertools.combinations(range(len(sets)), 2)
                    if sets[a] | sets[b] and 5 * len(sets[a] & sets[b]) >= 4 * len(sets[a] | sets[b])}
        self.assertEqual(set(matching_pairs(sets)), expected)
        self.assertIn((0, 1), expected)
        self.assertNotIn((0, 2), expected)

    def test_matching_a_headline_and_crossing_its_component_are_distinct(self):
        repeated = "One two three four five six seven eight nine ten eleven"
        pairs = [(repeated, "some unrelated text"), (repeated + " twelve", "my other title"),
                 ("my other title", "different last turn"), ("fixed side", "short text")]
        flags, info = compute(pairs, ["A", "B", "B", "C"], list(range(4)), {"first": [0, 1, 1, 1]})
        self.assertEqual(info["expanded_groups"], 2)
        self.assertEqual([r["first_direct"] for r in flags], [1, 1, 0, 0])
        self.assertEqual([r["first_intact"] for r in flags], [0, 0, 0, 1])
        self.assertFalse(shingles("a b c d e"))
        self.assertEqual(shingles("Ａ, b c d e F"), shingles("a b c d e f"))


class ExternalPreferenceTests(unittest.TestCase):
    @staticmethod
    def tree(name, question=None):
        common = dict(lang="en", deleted=False, synthetic=False, message_tree_id=name)
        q = question or f"Which answer fits tree {name}"
        root = dict(common, role="prompter", message_id=name, parent_id=None, rank=None, text=q)
        replies = [dict(common, role="assistant", message_id=f"{name}-{i}", parent_id=name, rank=i,
                        text=f"Response {name} ranked {i} gives a sufficiently long explanation for this example")
                   for i in range(2)]
        return [root, *replies]

    def test_order_is_fixed_by_tree_and_duplicate_prompts_stay_together(self):
        tree_ids = ("red", "green", "blue", "silver")
        messages = [m for tree in tree_ids for m in self.tree(tree, "A COMMON question here" if tree != "silver" else None)]
        first = make_pairs(messages)
        self.assertEqual(first, make_pairs(reversed(messages)))
        self.assertEqual(len(first), 2)
        duplicate_winner = min(tree_ids[:3], key=lambda t: hashed("pilot-external-trees-v1/", t))
        self.assertEqual({p[0] for p in first}, {duplicate_winner, "silver"})
        for tree, left, right, y in first:
            self.assertEqual(y, int(hashed("pilot-external-order-v1/", tree)[0] % 2 == 0))
            self.assertIn("ranked 0", left if y else right)
            self.assertEqual(left.split("\n")[0], right.split("\n")[0])

    def test_only_eligible_immediate_human_siblings_count(self):
        for field, value in (("lang", "de"), ("parent_id", "other"), ("synthetic", True), ("deleted", True),
                             ("text", "short"), ("rank", None)):
            with self.subTest(field=field):
                messages = self.tree("test")
                messages[1][field] = value
                self.assertFalse(make_pairs(messages))
        messages = self.tree("repeat-rank")
        self.assertFalse(make_pairs([*messages, messages[2]]))


if __name__ == "__main__":
    unittest.main()
