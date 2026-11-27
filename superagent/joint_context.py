"""Inductive categorical recommendation candidates with inferred attribute blocks."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from .field_interactions import InteractionTree, choose_pairs, rank_pairs
from .models import _clean_categories, _tabular_catboost
from .profile_history import discover_columns


def blocks(x):
    profile = discover_columns(x)
    context = []
    partitions = set()
    for j, col in enumerate(x.T):
        if j in profile:
            continue
        codes, levels = pd.factorize(col)
        signature = codes.tobytes()
        if len(levels) > 1 and signature not in partitions:
            context.append(j)
            partitions.add(signature)
    if not profile or not context:
        raise ValueError('No supported recurring profile and varying context')
    return profile, context


def tokens(x):
    return [''.join(f'{len(v)}:{v}' for v in row) for row in x]


class ContextTokenTree(InteractionTree):
    def transform(self, rows):
        x = _clean_categories(rows)
        base = super().transform(x)
        p = tokens(x[:, self.profile])
        c = tokens(x[:, self.context])
        conditional = [tokens(np.column_stack((p, x[:, j]))) for j in self.context]
        return np.column_stack((base, p, c, *conditional))

    def fit(self, rows, labels):
        x = _clean_categories(rows)
        self.profile, self.context = blocks(x)
        self.pairs = choose_pairs(rank_pairs(x, labels), self.params['pair_limit'])
        features = self.transform(x)
        self.classifier = _tabular_catboost(labels, 6, 3.)
        self.classifier.set_params(thread_count=6)
        self.classifier.fit(features, labels, cat_features=list(range(features.shape[1])))
        self.classes_ = self.classifier.classes_
        return self


class JointKernel:
    def fit(self, rows, labels):
        x = _clean_categories(rows)
        self.profile, self.context = blocks(x)
        self.encoder = OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1)
        self.x = self.encoder.fit_transform(x)
        self.classes_, y = np.unique(labels, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError('Joint response kernel requires binary labels')
        n = len(y)
        # The additive floor retains fields with weak marginal but useful conditional signal.
        self.weights = np.array([.08 + np.sqrt(max(0., mutual_info_score(col,y) - (len(np.unique(col))-1)/(2*n)))
                                 for col in self.x.T])
        self.prior = float(y.mean())
        gram = np.empty((n,n), dtype=np.float32, order='F')
        for i in range(0,n,128):
            gram[i:i+128] = self.kernel(self.x[i:i+128])
        gram.flat[::n+1] += 1.8
        with threadpool_limits(limits=6):
            factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
            self.coef = cho_solve(factor, (y-self.prior).astype(np.float32), check_finite=False)
        return self

    def similarity(self, query, columns):
        score = np.zeros((len(query),len(self.x)), dtype=np.float32)
        exact = np.ones(score.shape, dtype=bool)
        weights = self.weights[list(columns)]
        for j, weight in zip(columns, weights/weights.sum(), strict=True):
            matches = query[:,j,None] == self.x[None,:,j]
            score += matches * np.float32(weight)
            exact &= matches
        return score, exact

    def kernel(self, query):
        p, same_p = self.similarity(query,self.profile)
        c, same_c = self.similarity(query,self.context)
        return (.30*p**3 + .25*c**3 + .85*p**3*c**3 + .70*same_p*(.20+.80*c**4)
                + .65*same_c*(.10+.90*p**2))

    def predict_proba(self, rows):
        query = self.encoder.transform(_clean_categories(rows))
        with threadpool_limits(limits=6):
            scores = np.concatenate([self.kernel(query[i:i+128]) @ self.coef for i in range(0,len(query),128)])
        p = np.clip(self.prior + scores, 0., 1.)
        return np.column_stack((1-p,p))

    def predict(self, rows):
        return self.classes_[np.argmax(self.predict_proba(rows), axis=1)]
