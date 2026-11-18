import unittest
import numpy as np
from superagent.challengers import edit_features, CategoryCrosses
from superagent.models import search_candidates,probability_with_model


class ChallengersTests(unittest.TestCase):
    def test_swap_and_identical_pairs(self):
        pairs=[('red red x','red x red'),('what now','why now'),('','abc'),('Hi','hi')]
        reverse=[pair[::-1] for pair in pairs]
        self.assertEqual((edit_features(pairs)+edit_features(reverse)).nnz,0)
        self.assertGreater(edit_features(pairs[:1]).count_nonzero(),0)
        self.assertEqual(edit_features(pairs[-1:]).nnz,0)
        candidate=next(c for c in search_candidates('pair_text','challenger') if c.family=='edit_pair_C0.3')
        model,_=candidate.fit_predict(pairs*3,np.array([1,1,0,0]*3),reverse)
        classes,probs=probability_with_model('pair_text',model,pairs)
        np.testing.assert_allclose(probs[:,1],model.predict_proba(reverse)[:,0],atol=1e-8)

    def test_unseen_crosses_and_class_order(self):
        rows=np.array([['a','x'],['a','y'],['b','y'],['b','x']]*10)
        target=np.array([3,5,3,5]*10)
        model,_=next(c for c in search_candidates('categorical_tabular','challenger') if c.family.startswith('category_cross')).fit_predict(rows,target,rows)
        classes,probs=probability_with_model('categorical_tabular',model,[['new','x']])
        self.assertEqual(classes.tolist(),[3,5])
        self.assertAlmostEqual(probs.sum(),1.)
        self.assertEqual(CategoryCrosses().fit(rows).pairs_,[(0,1)])
