from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tools.crossfit_challenger import (combine, learn_weights, nominate, residual_directions,
                                      select, training_roles)
from superagent import ab_sentence


class CrossfitTests(unittest.TestCase):
    def test_screen_credits_reproducible_residual_direction_and_can_decline(self):
        groups = np.repeat([f"g{i}" for i in range(60)],2)
        y = np.tile([-1.,1.],60)
        x = np.column_stack((y,np.zeros(len(y))))
        positive = residual_directions(x,y,groups)
        opposite = residual_directions(-x,-y,groups)
        np.testing.assert_allclose(positive,opposite)
        score = float(positive[0] @ positive[1])
        self.assertGreater(score,0)
        self.assertEqual(nominate({"second":0,"first":score}),"first")
        self.assertIsNone(nominate({"second":-1,"first":0}))

    def test_roles_are_disjoint_and_invariant_to_names_and_row_permutation(self):
        g = np.repeat([f"g{i}" for i in range(180)],2)
        y = np.tile([0,1],180)
        result = training_roles(g,y)
        reverse = training_roles(g[::-1],y[::-1])
        self.assertEqual(sum(len(set(g[ix])) for ix in result.values()),len(set(g)))
        for key,ix in result.items():
            self.assertEqual(set(g[ix]),set(g[::-1][reverse[key]]))

    def test_oof_accepts_useful_expert_and_rejects_unstable_or_identical_outputs(self):
        y = np.tile([0,1],300)
        scores = np.column_stack((np.full(len(y),.55),np.full(len(y),.55),.1+.8*y))
        def records(prefix, values):
            return dict(scores=values,ties=np.zeros(len(y),dtype=bool),labels=y,
                        groups=np.array([prefix+str(i//2) for i in range(len(y))]))
        good = select({"a":records("a",scores),"b":records("b",scores)})
        self.assertEqual(good["chosen"],"agreement_weight")
        unstable = scores.copy()
        unstable[:,2] = 1-unstable[:,2]
        self.assertEqual(select({"a":records("a",scores),"b":records("b",unstable)})["chosen"],"baseline")
        ident = np.full_like(scores,.55)
        self.assertEqual(select({"a":records("a",ident),"b":records("b",ident)})["chosen"],"baseline")
        with self.assertRaises(ValueError):
            select({"a":records("a",scores),"b":records("a",scores)})

    def test_swap_and_duplicate_groups_do_not_give_extra_weight(self):
        p = np.array([[.8,.2,.8],[.1,.3,.9],[.4,.6,.15]])
        t = np.array([True,False,True])
        y = np.array([1,1,0])
        g = np.array(["g1","g2","g3"])
        w = learn_weights(p,t,y,g)
        np.testing.assert_allclose(w,learn_weights(np.repeat(p,2,axis=0),np.repeat(t,2),np.repeat(y,2),np.repeat(g,2)))
        np.testing.assert_allclose(combine(p,t,w),1-combine(1-p,~t,w))

    def test_run_cache_uses_only_raw_inputs_and_resets(self):
        invocations = []
        def encode(texts,*args,**kwargs):
            invocations.append(list(texts))
            return np.array([[len(t)] for t in texts]),len(texts)
        with patch.object(ab_sentence,"_encoder",return_value=(None,None)), patch.object(ab_sentence,"encode_frozen",side_effect=encode):
            with ab_sentence.reuse_frozen_vectors():
                ab_sentence._pinned_vectors(["abcd","abc"],Path("pin"),256)
                result,count=ab_sentence._pinned_vectors(["abc","f","abcd"],Path("pin"),256)
                self.assertEqual((result.ravel().tolist(),count),([3,1,4],1))
            ab_sentence._pinned_vectors(["abc"],Path("pin"),256)
        self.assertEqual(invocations,[["abcd","abc"],["f"],["abc"]])
