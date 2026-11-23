import math
import os
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone

from scripts.check_b_word_surprisal import corpus_scores
from superagent.ab_methods import additions, PREFERENCE_CATALOG_ENV, TEXT_DEEP, TEXT_FAMILIARITY
from superagent.ab_sentence import FamiliarityPreferenceFeatures
from superagent.external_assets import model_resources
from superagent.word_familiarity import WordFamiliarity


class WordFamiliarityTests(unittest.TestCase):
    titles = ['red blue red', 'blue red blue', 'red red blue', 'blue moon!']

    def test_unseen_queries_use_the_original_training_only_estimator(self):
        corpus = WordFamiliarity().fit([*self.titles, self.titles[0]])
        original, _ = corpus_scores(self.titles)
        for text in ['', 'ice ice ice', 'RED blue?', 'one blue red!']:
            self.assertAlmostEqual(corpus.score(text), original(text), places=12)
        self.assertEqual(len(corpus.title_digests_), len(self.titles))
        self.assertTrue(all(type(d) is bytes and len(d) == 32 for d in corpus.title_digests_))

    def test_a_training_title_is_scored_with_its_own_counts_removed(self):
        corpus = WordFamiliarity().fit(self.titles)
        original, _ = corpus_scores(self.titles[1:])
        self.assertEqual(corpus.vocabulary_, {'red', 'blue'})
        self.assertAlmostEqual(corpus.score(self.titles[0]), original(self.titles[0]), places=12)
        self.assertTrue(math.isfinite(WordFamiliarity().fit(['a a a']).score('a a a')))
        rows = [(self.titles[0], self.titles[1]), (self.titles[1], self.titles[0]), ('', '')]
        a, b, c = corpus.differences(rows)
        self.assertEqual(a, -b)
        self.assertEqual(c, 0)

    def test_scaled_feature_is_antisymmetric_and_clone_has_the_same_resource(self):
        f = clone(FamiliarityPreferenceFeatures('pair_text'))
        rows = [(self.titles[0], self.titles[1]), (self.titles[1], self.titles[0]),
                (self.titles[2], self.titles[3]), (self.titles[3], self.titles[3])]
        f.fit(rows)
        with patch('superagent.ab_sentence.encode', return_value=(np.zeros((8, 2)), 8)):
            blocks, _ = f.semantic_blocks(rows, [t for pair in rows for t in pair])
        delta = blocks[-1].toarray().ravel()
        self.assertAlmostEqual(np.mean(delta ** 2), 1)
        self.assertEqual(delta[0], -delta[1])
        self.assertEqual(delta[-1], 0)
        self.assertEqual([r['name'] for r in model_resources(f)[0]], ['minilm_l12'])

    def test_only_the_explicit_verified_pair_catalog_exposes_the_new_method(self):
        def names(kind, flavor, ready):
            with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: flavor}), patch(
                    'superagent.ab_sentence.verified_directory', return_value=ready):
                return [m.family for m in additions(kind)]
        prior = names('pair_text', 'deep', True)
        both = names('pair_text', 'deep-familiarity', True)
        self.assertIn(TEXT_DEEP, prior)
        self.assertNotIn(TEXT_FAMILIARITY, prior)
        self.assertEqual(prior, [m for m in both if m != TEXT_FAMILIARITY])
        self.assertNotIn(TEXT_FAMILIARITY, names('single_text', 'deep-familiarity', True))
        self.assertNotIn(TEXT_FAMILIARITY, names('pair_text', 'deep-familiarity', None))
