import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from superagent import ab_sentence


class TestExactSentenceReuse(TestCase):
    def test_order_duplicates_pool_and_ownership(self):
        batches=[]
        def encode(texts,*_,prefix,pool):
            batches.append((tuple(texts),prefix,pool))
            return np.array([[ord(t)] for t in texts],dtype=np.float32),len(set(texts))
        ab_sentence._recent_batch.cache_clear()
        self.addCleanup(ab_sentence._recent_batch.cache_clear)
        with (patch.dict(os.environ,{"SUPERAGENT_FROZEN_BATCH_REUSE":"1"}),
              patch.object(ab_sentence,"_encoder",return_value=(None,None)),
              patch.object(ab_sentence,"encode_frozen",side_effect=encode)):
            def run(texts,pool="mean"):
                return ab_sentence._batch("verified-root",128,texts,prefix="",pool=pool)
            values,tokens=run(["a","b","a"])
            np.testing.assert_array_equal(values[:,0],[97,98,97])
            self.assertEqual(tokens,2)
            values[0,0]=0
            np.testing.assert_array_equal(run(["a","a","b"])[0][:,0],[97,97,98])
            self.assertEqual(len(batches),1)
            run(["b","a"])
            run(["a","b"],"cls")
            self.assertEqual(len(batches),3)
