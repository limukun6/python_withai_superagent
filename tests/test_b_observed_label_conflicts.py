import itertools
import unittest
from collections import Counter

from scripts.audit_b_released_label_conflicts import categorical_key, characterize, preference_graph, summary


class ConflictBoundsTests(unittest.TestCase):
    def test_exact_rows_and_swapped_direction_have_separate_meanings(self):
        rows=[('a','b'),('b','a'),('a','b'),('x','x')]
        result,_,_=characterize(rows,[1,1,0,0],True)
        self.assertEqual(result['ordered']['minimum_wrong'],1)
        self.assertEqual(result['unordered']['minimum_wrong'],1)
        self.assertEqual(result['preference_graph']['same_string_rows'],1)
        other,_,_=characterize([('a','b'),('b','a')],[1,1],True)
        self.assertEqual(other['ordered']['minimum_wrong'],0)
        self.assertEqual(other['unordered']['minimum_wrong'],1)
        self.assertNotEqual(categorical_key([1,'x']),categorical_key(['1','x']))

    def test_opposite_votes_and_edge_disjoint_cycle_losses_add(self):
        observed={('a','b'):Counter({1:4,0:1}),('b','c'):Counter({1:3,0:1}),('a','c'):Counter({0:5,1:1})}
        self.assertEqual(summary(observed)['minimum_wrong'],3)
        graph=preference_graph(observed)
        self.assertEqual((graph['closed_triangles'],graph['edge_disjoint_cycles']),(1,1))
        self.assertEqual(graph['additional_mandatory_errors_for_one_global_order'],2)

    def test_certificate_never_exceeds_exact_minimum_in_all_four_title_fixtures(self):
        titles='abcd'
        pairs=list(itertools.combinations(titles,2))
        orders=[{name:idx for idx,name in enumerate(o)} for o in itertools.permutations(titles)]
        for outcomes in itertools.product((-1,0,1),repeat=len(pairs)):
            counts={k:Counter({1:1+max(s,0),0:1+max(-s,0)}) for k,s in zip(pairs,outcomes,strict=True)}
            graph=preference_graph(counts)
            floor=summary(counts)['minimum_wrong']+graph['additional_mandatory_errors_for_one_global_order']
            # Enumerate fixture orderings only to independently verify the bound.
            exact=min(sum(c[int(order[a]>order[b])] for (a,b),c in counts.items()) for order in orders)
            self.assertLessEqual(floor,exact)
