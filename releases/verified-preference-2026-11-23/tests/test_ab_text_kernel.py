import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from superagent.ab_sentence import KERNEL_DIMENSIONS, KERNEL_SAMPLE_LIMIT, KernelSentenceFeatures, SentenceFeatures


def example_embeddings(texts):
    vectors = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
        dense = np.random.default_rng(seed).normal(size=384).astype(np.float32)
        vectors.append(dense/np.linalg.norm(dense))
    return np.stack(vectors),sum(len(text.split()) for text in set(texts))


class TestKernelTextFeatures(unittest.TestCase):
    @patch('superagent.ab_sentence.encode',side_effect=example_embeddings)
    def test_original_columns_preserved_and_pair_features_reverse(self,_):
        rows = [('a quiet blue bridge','a bright green bridge'),('another blue bridge','another green bridge')]*2
        reverse = [(b,a) for a,b in rows]
        plain = SentenceFeatures('pair_text').fit(rows).transform(rows)
        features = KernelSentenceFeatures('pair_text').fit(rows)
        actual = features.transform(rows)
        self.assertEqual(actual.shape[1],plain.shape[1]+KERNEL_DIMENSIONS)
        self.assertEqual((actual[:,:plain.shape[1]]-plain).nnz,0)
        np.testing.assert_allclose((actual+features.transform(reverse)).toarray(),0,atol=1e-6)
        self.assertEqual((actual[:,-KERNEL_DIMENSIONS:]).shape,(len(rows),512))
        self.assertGreater((actual[:,-KERNEL_DIMENSIONS:]).nnz,0)

    @patch('superagent.ab_sentence.encode',side_effect=example_embeddings)
    def test_sample_is_train_only_capped_order_and_label_independent(self,encode):
        rows = [f'common text example {i}' for i in range(KERNEL_SAMPLE_LIMIT+19)]
        forward = KernelSentenceFeatures('single_text').fit(rows,[0]*len(rows))
        sample = list(encode.call_args.args[0])
        self.assertEqual(len(sample),KERNEL_SAMPLE_LIMIT)
        self.assertTrue(set(sample)<=set(rows))
        backward = KernelSentenceFeatures('single_text').fit(rows[::-1],[1]*len(rows))
        self.assertEqual(encode.call_args.args[0],sample)
        self.assertEqual(forward.median_squared_distance_,backward.median_squared_distance_)
        self.assertAlmostEqual(np.exp(-forward.kernel_.gamma*forward.median_squared_distance_),.5)
        np.testing.assert_array_equal(forward.kernel_.random_weights_,backward.kernel_.random_weights_)
        before = forward.kernel_.random_weights_.copy()
        self.assertTrue(np.isfinite(forward.transform(['previously unseen text']).data).all())
        np.testing.assert_array_equal(before,forward.kernel_.random_weights_)

    @patch('superagent.ab_sentence.encode',side_effect=example_embeddings)
    def test_no_distinct_distances_has_a_finite_fallback_and_accounted_tokens(self,_):
        repeated = ['same shared text']*4
        features = KernelSentenceFeatures('single_text').fit(repeated)
        self.assertIsNone(features.median_squared_distance_)
        self.assertEqual(features.kernel_.gamma,1.)
        self.assertEqual(features.usage_['bandwidth_input_tokens'],3)
        self.assertEqual(features.usage_['input_tokens'],3)
        self.assertTrue(np.isfinite(features.transform(repeated).data).all())
        self.assertEqual(features.usage_['input_tokens'],6)
