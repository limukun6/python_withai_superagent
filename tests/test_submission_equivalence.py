from copy import deepcopy
import unittest

from tools.submission_equivalence import equivalent_csv_recipe
from scripts.sa5_semantic_cases import output


class SubmissionEquivalenceTests(unittest.TestCase):
    def recipe(self):
        return output([dict(name='Id',kind='input_id',field='record_id'),dict(name='Label',kind='label'),
                       dict(name='Probability',kind='probability',class_label='1')],delimiter=',',encoding='utf-8')

    def test_identity_mapping_and_explicit_unique_default_are_equivalent(self):
        expected=self.recipe();actual=deepcopy(expected)
        actual['columns'][0]['unique']=True
        actual['columns'][1]['mapping']={'0':'0','1':'1'}
        self.assertTrue(equivalent_csv_recipe(actual,expected,['0','1']))
        actual['columns'][1]['mapping']={'0':0,'1':1}
        self.assertTrue(equivalent_csv_recipe(actual,expected,['0','1']))

    def test_all_visible_submission_changes_remain_inequivalent(self):
        expected=self.recipe()
        edits=[lambda r:r.update(delimiter=';'),lambda r:r.update(encoding='utf-8-sig'),
               lambda r:r.update(header=False),lambda r:r['columns'][0].update(field='other_id'),
               lambda r:r['columns'][0].update(unique=False),lambda r:r['columns'][1].update(name='OtherHeader'),
               lambda r:r['columns'][1].update(mapping={'0':'1','1':'0'}),
               lambda r:r['columns'][2].update(class_label='0')]
        for edit in edits:
            actual=deepcopy(expected);edit(actual)
            self.assertFalse(equivalent_csv_recipe(actual,expected,['0','1']))

    def test_typed_json_values_cannot_collapse_to_the_same_csv_label(self):
        expected=self.recipe();actual=deepcopy(expected)
        actual['columns'][1]['mapping']={'0':0,'1':'0'}
        self.assertFalse(equivalent_csv_recipe(actual,expected,['0','1']))

    def test_partial_mapping_is_not_a_valid_equivalent_recipe(self):
        expected=self.recipe();actual=deepcopy(expected)
        actual['columns'][1]['mapping']={'0':'0'}
        self.assertFalse(equivalent_csv_recipe(actual,expected,['0','1']))


if __name__=='__main__':unittest.main()
