import unittest

import numpy as np

from tools.encoder_weight_headroom import binary_choices, fit_weight


class EncoderWeightTests(unittest.TestCase):
    def test_ties_and_open_intervals_are_both_checked(self):
        p=np.asarray([.75,.25,.5,.875])
        q=np.asarray([.25,.75,.5,.625])
        weights,predict=binary_choices(p,q)
        self.assertTrue({0.,.25,.5,.75,1.}<=set(weights))
        for w in weights:
            np.testing.assert_array_equal(predict(w),w*p+(1-w)*q>.5)
        np.testing.assert_array_equal(predict(.5),[0,0,0,1])

    def test_repeating_inner_rows_does_not_shift_a_balanced_optimum(self):
        y=np.asarray([1,0])
        probs=np.asarray([[.8,.2],[.8,.2]])
        one=fit_weight([probs],[np.arange(2)],y,np.arange(2))
        many=fit_weight([probs,probs[:1]],[np.arange(2),np.array([0])],y,np.arange(2))
        self.assertAlmostEqual(one['raw'],.5)
        self.assertAlmostEqual(one['raw'],many['raw'])
        self.assertEqual(many['unique_rows'],2)
