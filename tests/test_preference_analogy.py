import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from superagent.analogy_review import blend_counts, promoted
from superagent.preference_analogy import AnalogyIndex,WeightedProbabilityVote


class ToyModel(ClassifierMixin,BaseEstimator):
    def __init__(self,probability,classes):
        self.probability=np.asarray(probability)
        self.classes_=np.asarray(classes)

    def fit(self,rows,labels):
        return self

    def predict_proba(self,rows):
        return np.tile(self.probability,(len(rows),1))


class Comparison(unittest.TestCase):
    def test_derived_preference_vote_is_symmetric_and_does_not_count_a_related_group_twice(self):
        vectors={'a':[1.,0,1],'b':[0,1,1],'c':[1,0,2],'d':[0,1,2]}
        def provider(texts):
            return np.asarray([vectors[t] for t in texts])
        duplicate=[('a','b'),('a','b'),('d','c')]
        short=AnalogyIndex().fit([duplicate[0],duplicate[-1]],[1,0],['g','h'],provider)
        full=AnalogyIndex().fit(duplicate,[1,1,0],['g','g','h'],provider)
        rows=[('a','b'),('b','a'),('a','a')]
        output=full.predict_probability(rows,provider)
        np.testing.assert_allclose(output,short.predict_probability(rows,provider))
        self.assertGreater(output[0],.5)
        self.assertAlmostEqual(output[0]+output[1],1.)
        self.assertEqual(output[2],.5)

    def test_ordered_probabilities_match_after_class_reversal(self):
        first=Pipeline([('classifier',ToyModel([.3,.7],[0,1]))])
        second=Pipeline([('classifier',ToyModel([.1,.9],[1,0]))])
        ensemble=WeightedProbabilityVote('pair_text',[first,second],[0,1],[.75,.25])
        np.testing.assert_allclose(ensemble.predict_proba([('x','y')]),[[.45,.55]])

    def test_reject_mismatched_rows_and_hold_a_gain_to_confirmation(self):
        truth=np.array([1,0])
        a=dict(row_key='source',labels=truth,probabilities=np.array([[.6,.4],[.8,.2]]))
        b=dict(a,probabilities=np.array([[.2,.8],[.1,.9]]))
        self.assertEqual(blend_counts(a,b,truth,[0.,.5,1.]),[1,1,1])
        with self.assertRaisesRegex(ValueError,'aligned'):
            blend_counts(a,dict(b,row_key='different'),truth,[.5])
        self.assertFalse(promoted(1,[dict(rows=1000,gain=12),dict(rows=1000,gain=-1)]))
        self.assertTrue(promoted(1,[dict(rows=1000,gain=6),dict(rows=1000,gain=0)]))
