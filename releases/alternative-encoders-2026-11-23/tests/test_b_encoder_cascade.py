import unittest
from unittest.mock import patch

import numpy as np

from scripts.b_encoder_cascade import choose, positive_labels, predict


def fold(a, b, y):
    return dict(scores={'a':np.array(a),'b':np.array(b)},labels=np.array(y),tie_left=np.array([True]*len(y)))


class CascadePolicyTests(unittest.TestCase):
    def test_new_evidence_can_reject_skipping(self):
        prior=fold([.9,.1,.51,.49],[.85,.15,.85,.15],[1,0,1,0])
        changed=fold([.55,.45,.55,.45],[.05,.95,.05,.95],[0,1,0,1])
        accepted=choose(prior,prior)
        rejected=choose(prior,changed)
        self.assertEqual(accepted['first'],'a')
        self.assertTrue(accepted['adopted'])
        self.assertEqual((accepted['proposed'],accepted['threshold']),(0.,0.))
        self.assertFalse(rejected['adopted'])
        self.assertEqual((rejected['proposed'],rejected['threshold']),(0.,1.))

    def test_second_encoder_receives_only_low_margin_rows(self):
        rows=[('a','w'),('b','x'),('c','y'),('d','z')]
        calls=[]
        def evaluate(model, batch):
            calls.append((model,batch))
            values={'a':[.9,.51,.35,.1],'b':[.2,.9]}
            return np.array(values[model]),dict(rows=len(batch),input_tokens=len(batch)*3)
        with patch('scripts.b_encoder_cascade.evaluate',side_effect=evaluate):
            selected,base,mask,usage=predict({'a':'a','b':'b'},rows,{'first':'a','second':'b'},.2)
        self.assertEqual(calls,[('a',rows),('b',rows[1:3])])
        np.testing.assert_array_equal(mask,[False,True,True,False])
        np.testing.assert_array_equal(base,[1,1,0,0])
        np.testing.assert_array_equal(selected,[1,0,1,0])
        self.assertEqual(usage['skips'],2)

    def test_tie_respects_pair_order(self):
        out=positive_labels(np.array([.5,.5,.4,.6]),np.array([True,False,True,False]))
        np.testing.assert_array_equal(out,[1,0,0,1])


if __name__=='__main__':
    unittest.main()
