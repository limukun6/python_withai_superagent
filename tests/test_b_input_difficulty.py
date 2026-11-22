import unittest

from scripts.audit_b_released_input_difficulty import categorical_strata, graph_counts, title_strata, word_distance


class InputDifficultyTests(unittest.TestCase):
    def test_directional_four_cycle_and_acyclic_connected_component(self):
        pairs=[('a','b'),('b','c'),('c','d'),('d','a'),('x','y')]
        graph=graph_counts(pairs,[1,1,1,1,1])
        self.assertEqual(graph['cyclic_strong_components'],1)
        self.assertEqual(graph['titles_on_directed_cycles'],4)
        self.assertEqual(graph['edges_inside_cyclic_components'],4)
        self.assertEqual(graph['components_with_one_comparison'],1)
        self.assertTrue(graph_counts(pairs,[1,1,1,0,1])['acyclic'])

    def test_word_edits_and_group_boundaries(self):
        self.assertEqual(word_distance('你好，世界','你好 世界!'),0)
        self.assertEqual(word_distance('a b','a X b'),1)
        self.assertEqual(word_distance('A b c','c b a'),2)
        values=title_strata([('A  X','q')],[('a x','b'),('b','c'),('z','w')])
        self.assertEqual(values['titles_in_optimization'],['1','0','0'])
        self.assertEqual(values['component_comparisons'],['multiple','multiple','one'])

    def test_new_combinations_of_existing_fields(self):
        a,b=[0]*21,[1]*21
        values=categorical_strata([a,b],[a,[1]+a[1:],[2]+a[1:]])
        self.assertEqual(values['unseen_field_value'],['none','none','one_or_more'])
        self.assertEqual(values['unseen_field_combinations'],['0','6_plus','6_plus'])
