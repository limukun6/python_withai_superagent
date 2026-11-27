import unittest

import numpy as np

from superagent.ab_methods import CONTEXT
from superagent.ab_run import registered_capabilities
from superagent.joint_context import blocks, tokens
from superagent.models import search_candidates
from tests.test_ab_scheduler import recorded


class RecurringContextTests(unittest.TestCase):
    def test_catalog_requires_support_binary_domain_and_permitted_model(self):
        spec = dict(recorded()['task'],kind='categorical_tabular',input_schema={})
        self.assertNotIn(CONTEXT, registered_capabilities(spec,800))
        spec['metric'] = dict(spec['metric'],labels=[0,1])
        cap = registered_capabilities(spec,800,repeated_context=True)[CONTEXT]
        self.assertEqual(cap['estimator'],'catboost')
        self.assertEqual(cap['classifier_fits'],1)
        self.assertEqual(sum(c.family==CONTEXT for c in search_candidates('categorical_tabular','all')),1)
        self.assertNotIn(CONTEXT, registered_capabilities(dict(spec,method_restrictions=['linear_only']),
                                                          800,repeated_context=True))
        spec['metric'] = dict(spec['metric'],labels=[0,1,2])
        self.assertNotIn(CONTEXT, registered_capabilities(spec,800,repeated_context=True))

    def test_unsupervised_discovery_needs_recurrence_and_variation(self):
        rng = np.random.default_rng(145)
        shuffled = rng.integers(12,size=(600,7)).astype(str)
        with self.assertRaises(ValueError):
            blocks(shuffled)
        entity = rng.integers(9,size=(40,6))
        x = np.column_stack((np.repeat(entity,15,axis=0),rng.integers(4,size=600))).astype(str)
        p,c = blocks(x)
        self.assertEqual(set(p),set(range(6)))
        self.assertEqual(c,[6])

    def test_tokens_preserve_field_boundaries_and_unseen_values(self):
        self.assertEqual(len(set(tokens([['a','bc'],['ab','c'],['','abc'],[':', '東京']]))),4)
