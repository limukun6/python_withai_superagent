import unittest
from unittest.mock import patch

import numpy as np

from superagent.ab_complement import PreferenceProbabilityVote, eligible
from superagent.metrics import MetricSpec


class ComplementScopeTests(unittest.TestCase):
    def test_scope_and_two_refit_budget(self):
        schema={'kind':'pair_text','input_schema':{'pair_semantics':'preference'}}
        caps={str(i):dict(classifier_fits=1,refit_classifier_fits=1,supports_probabilities=True) for i in range(4)}
        metric=MetricSpec('accuracy',(0,1))
        self.assertTrue(eligible(schema,caps,metric,8))
        self.assertFalse(eligible(schema,caps,metric,7))
        self.assertFalse(eligible(dict(schema,input_schema={'pair_semantics':'symmetric'}),caps,metric,8))
        self.assertFalse(eligible(schema,dict(list(caps.items())[:3]),metric,8))

    def test_class_order_swaps_identical_text_and_exact_ties(self):
        def score(kind,model,rows):
            self.assertTrue(all(a<=b for a,b in rows))
            v=np.array([model]*len(rows))
            return np.array([1,0]),np.column_stack([v,1-v])
        with patch('superagent.ab_complement.probability_with_model',side_effect=score):
            rows=[('a','b'),('b','a'),('a','a')]
            strong=PreferenceProbabilityVote('pair_text',[.7,.9],np.array([0,1]))
            np.testing.assert_allclose(strong.predict_proba(rows)[:,1],[.8,.2,.5])
            tie=PreferenceProbabilityVote('pair_text',[.25,.75],np.array([0,1]))
            self.assertEqual(tie.predict(rows).tolist(),[1,0,0])
