import unittest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from superagent.ab_preference_output import CanonicalPreferencePipeline


class Fixed(ClassifierMixin,BaseEstimator):
    def __init__(self, positive=.7, reversed_classes=False):
        self.positive=positive
        self.reversed_classes=reversed_classes
        self.classes_=np.asarray([1,0] if reversed_classes else [0,1])
        self.calls=0

    def fit(self, X, y=None):
        raise AssertionError('Frozen inference cannot train')

    def predict_proba(self, X):
        self.calls+=1
        return np.tile([self.positive if k==1 else 1.-self.positive for k in self.classes_],(len(X),1))


class CanonicalOutputTests(unittest.TestCase):
    def test_permutation_class_order_and_one_call(self):
        pairs=[('a','b'),('b','a'),('相同','相同'),('','Ω')]
        for reverse in (False,True):
            final=Fixed(reversed_classes=reverse)
            model=CanonicalPreferencePipeline([('head',final)])
            p=model.predict_proba(pairs)
            np.testing.assert_allclose(p[:,list(model.classes_).index(1)],[.7,.3,.5,.7])
            self.assertEqual(final.calls,1)
            np.testing.assert_array_equal(model.predict(pairs),[1,0,0,1])
            self.assertEqual(final.calls,2)

    def test_exact_ties_choose_opposite_directions_for_distinct_strings(self):
        final=Fixed(positive=.5)
        model=CanonicalPreferencePipeline([('head',final)])
        pairs=[('甲','乙'),('乙','甲'),('x','x')]
        labels=model.predict(pairs)
        self.assertEqual(int(labels[0])+int(labels[1]),1)
        self.assertEqual(labels[2],0)
        np.testing.assert_array_equal(model.predict_proba(pairs),np.full((3,2),.5))
        self.assertEqual(model.predict_proba([]).shape,(0,2))
        self.assertEqual(final.calls,2)

    def test_refuses_invalid_rows_and_unusable_probabilities(self):
        model=CanonicalPreferencePipeline([('head',Fixed(positive=np.nan))])
        for rows in ([('a',1)],['ab'],[('a','b')]):
            with self.assertRaises(ValueError):
                model.predict(rows)
