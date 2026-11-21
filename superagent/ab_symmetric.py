"""One supervised fit on features invariant to the order of the two input texts."""
from functools import partial

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .models import Candidate


CHANNELS = {'unordered_word': ('word',), 'unordered_char': ('char',),
            'unordered_word_char': ('word', 'char')}


class UnorderedPairFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, channels=('word',), bins=32768):
        self.channels = channels
        self.bins = bins

    def fit(self, rows, y=None):
        self.vectorizers_ = []
        for channel in self.channels:
            options = dict(analyzer='char_wb', ngram_range=(3, 5)) if channel == 'char' else dict(
                analyzer='word', ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
            self.vectorizers_.append(HashingVectorizer(**options, n_features=self.bins,
                                                       alternate_sign=False, norm='l2', dtype=np.float64))
        return self

    def transform(self, rows):
        left, right = zip(*rows)
        blocks = []
        for vectorizer in self.vectorizers_:
            a, b = vectorizer.transform(left), vectorizer.transform(right)
            blocks.extend((abs(a - b), a.multiply(b)))
        features = hstack(blocks, format='csr')
        features.sort_indices()
        return features


def fit_symmetric_pair(rows, labels, query, *, channels, bins, c):
    model = Pipeline([('unordered_pair', UnorderedPairFeatures(channels, bins)),
                      ('classifier', LogisticRegression(C=c, solver='liblinear', max_iter=500, random_state=42))])
    model.fit(rows, labels)
    return model, model.predict(query)


def candidates():
    return [Candidate(name, dict(representation=name, hash_bins=32768, C=1.0),
                      partial(fit_symmetric_pair, channels=channels, bins=32768, c=1.0))
            for name, channels in CHANNELS.items()]
