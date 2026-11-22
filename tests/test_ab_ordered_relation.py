import unittest

import numpy as np

from superagent.ab_catalog import common_catalog
from superagent.ab_methods import RELATION, additions
from superagent.schema import TextTableRows


class OrderedRelationTests(unittest.TestCase):
    def test_catalogue_exposes_interactions_only_for_the_ordered_two_text_contract(self):
        columns = [dict(name='premise',type='text'),dict(name='hypothesis',type='text')]
        metric = dict(name='accuracy',labels=['a','b','c'],direction='maximize')
        def names(schema):
            task = dict(contract_version=1,objective='classification',metric=metric,
                        kind='text_tabular',input_schema=schema)
            return [c['id'] for c in common_catalog(task,train_rows=32)]
        schema = dict(pair_semantics='relation',modality='pair_text',columns=columns)
        old = names(dict(modality='tabular',columns=columns))
        self.assertEqual(names(schema), old+[RELATION])
        for change in (dict(pair_semantics='symmetric'),dict(columns=columns[:1]),dict(columns=None),
                       dict(columns=columns+[dict(name='value',type='numeric')])):
            self.assertNotIn(RELATION,names(dict(schema,**change)))

    def test_swapping_preserves_overlap_and_changes_the_ordered_feature_block(self):
        texts = np.array([['red apples grow','green pears grow'],['short green grass','red apples grow'],
                          ['red leaves fall','short green grass'],['green pears grow','red leaves fall'],
                          ['short green grass','red leaves fall'],['green pears grow','short green grass']],dtype=object)
        def rows(values):
            return TextTableRows(values,(),('first','second'),(0,1))
        candidate = next(c for c in additions('text_tabular') if c.family==RELATION)
        model,predicted = candidate.fit_predict(rows(texts),np.array(['a','b','c','a','b','c']),rows(texts))
        self.assertEqual(len(predicted),len(texts))
        self.assertEqual(set(model.classes_),{'a','b','c'})
        blocks = dict(model.named_steps['features'].transformer_list)
        forward, reverse = rows(texts),rows(texts[:,::-1])
        self.assertEqual((blocks['overlap'].transform(forward)!=blocks['overlap'].transform(reverse)).nnz,0)
        self.assertGreater((blocks['ordered'].transform(forward)!=blocks['ordered'].transform(reverse)).nnz,0)
