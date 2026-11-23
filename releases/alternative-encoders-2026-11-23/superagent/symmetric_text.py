"""Training-only lexical pair features whose output is invariant to swapping sides."""
from functools import partial

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _pairs(rows):
    values = list(rows)
    for row in values:
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 2 or any(not isinstance(text, str) for text in row):
            raise ValueError('Expected two string texts per pair')
    return values


class SymmetricTfidf(TransformerMixin, BaseEstimator):
    """Fit shared vocabularies on training texts; encode |left-right| and left*right."""
    def __init__(self, representation='word', max_features=20000):
        self.representation = representation
        self.max_features = max_features

    def fit(self, rows, y=None):
        if self.representation not in ('word', 'word_char'):
            raise ValueError('Symmetric representation must be word or word_char')
        pairs = _pairs(rows)
        if not pairs:
            raise ValueError('Cannot fit an empty collection of pairs')
        texts = [text for pair in pairs for text in pair]
        self.vectorizers_ = [TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
            max_features=self.max_features, sublinear_tf=True, dtype=np.float64)]
        if self.representation == 'word_char':
            self.vectorizers_.append(TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                max_features=self.max_features, sublinear_tf=True, dtype=np.float64))
        for vectorizer in self.vectorizers_:
            vectorizer.fit(texts)
        return self

    def transform(self, rows):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'vectorizers_')
        pairs = _pairs(rows)
        if not pairs:
            from scipy.sparse import csr_matrix
            return csr_matrix((0, sum(2 * len(v.vocabulary_) for v in self.vectorizers_)))
        left, right = zip(*pairs)
        blocks = []
        for vectorizer in self.vectorizers_:
            a, b = vectorizer.transform(left), vectorizer.transform(right)
            blocks.extend((abs(a - b), a.multiply(b)))
        matrix = hstack(blocks, format='csr')
        matrix.sort_indices()
        return matrix


def fit_symmetric_pair(rows, labels, query, *, representation='word', c=.3):
    model = Pipeline([
        ('symmetric_tfidf', SymmetricTfidf(representation=representation)),
        ('classifier', LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])
    model.fit(rows, labels)
    return model, model.predict(query)


def candidates():
    from .models import Candidate
    return [Candidate(f'symmetric_{representation}_C{c}', dict(C=c,
        representation='symmetric_' + representation + '_absdiff_product', swap_augmentation=False,
        max_features_per_vectorizer=20000), partial(fit_symmetric_pair, representation=representation, c=c))
        for representation in ('word', 'word_char') for c in (.3, 3.)]
