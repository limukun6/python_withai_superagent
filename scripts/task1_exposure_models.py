"""Two inductive candidates for a categorical table with repeated profiles."""
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.models import _clean_categories


def context_axis(x, y, context):
    return max(context, key=lambda col: mutual_info_score(x[:, col], y))


class ExposureTree(ContextTokenTree):
    def fit(self, rows, labels):
        x = _clean_categories(rows)
        profile, context = blocks(x)
        names, people = np.unique(tokens(x[:, profile]), return_inverse=True)
        events, occasions = np.unique(tokens(x[:, context]), return_inverse=True)
        self.people = {name: i for i, name in enumerate(names)}
        self.occasions = {name: i for i, name in enumerate(events)}
        self.counts = np.zeros((len(names), len(events)))
        np.add.at(self.counts, (people, occasions), 1.)
        background = self.counts.sum(axis=0) + 1.
        background /= background.sum()
        n_clusters = min(8, len(names), len(events))
        with threadpool_limits(limits=2):
            hist = self.counts / self.counts.sum(axis=1, keepdims=True)
            initial = KMeans(n_clusters=n_clusters, n_init=5, random_state=301).fit_predict(np.sqrt(hist))
            weights = np.eye(n_clusters)[initial]
            for _ in range(25):
                self.log_prior = np.log((weights.sum(axis=0) + 1) / (len(names) + n_clusters))
                freq = weights.T @ self.counts + 20. * background
                self.log_frequency = np.log(freq / freq.sum(axis=1, keepdims=True))
                weights = softmax(self.counts @ self.log_frequency.T + self.log_prior, axis=1)
        return super().fit(rows, labels)

    def transform(self, rows):
        x = _clean_categories(rows)
        base = super().transform(x)
        people = np.array([self.people.get(p, -1) for p in tokens(x[:, self.profile])])
        events = tokens(x[:, self.context])
        occasions = np.array([self.occasions.get(c, -1) for c in events])
        known = people >= 0
        scores = np.tile(self.log_prior, (len(x), 1))
        with threadpool_limits(limits=2):
            scores[known] += (self.counts @ self.log_frequency.T)[people[known]]
        same = known & (occasions >= 0)
        scores[same] -= self.counts[people[same], occasions[same], None] * self.log_frequency.T[occasions[same]]
        group = np.argmax(scores, axis=1).astype(str)
        return np.column_stack((base, group, tokens(np.column_stack((group, events)))))


class PooledPreferences:
    """Regularized main and interaction effects, including sparse profile-specific terms."""
    def _coded(self, x, fit):
        raw = np.column_stack((x, tokens(x[:, self.profile]), tokens(x[:, self.context])))
        if fit:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
            self.encoder.fit(raw)
        return self.encoder.transform(raw) + 1

    def fit(self, rows, labels):
        x = _clean_categories(rows)
        self.profile, self.context = blocks(x)
        selected = context_axis(x, labels, self.context)
        encoded = self._coded(x, True)
        p, c = x.shape[1], x.shape[1] + 1
        fields = [*self.profile, *self.context]
        terms = {(j,): 1. for j in fields + [p, c]}
        terms.update({tuple(sorted(pair)): .5 for pair in combinations(fields, 2)})
        terms.update({(j, p): .75 for j in self.context})
        terms.update({(j, c): .35 for j in self.profile})
        for j, k in combinations(self.profile, 2):
            terms[tuple(sorted((j, k, selected)))] = .23
        for j in self.profile:
            for k in self.context:
                if k != selected:
                    terms[tuple(sorted((j, selected, k)))] = .3
        self.terms = tuple(terms)
        self.scales = np.asarray(list(terms.values()))
        self.radices = [len(levels) + 1 for levels in self.encoder.categories_]
        self.basis = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        matrix = self.basis.fit_transform(self._crosses(encoded))
        self.weights = np.repeat(self.scales, [len(a) for a in self.basis.categories_])
        self.classifier = LogisticRegression(C=1.3, solver='liblinear', dual=True, random_state=301, max_iter=500)
        self.classifier.fit(self._scaled(matrix), labels)
        self.classes_ = self.classifier.classes_
        return self

    def _crosses(self, encoded):
        parts = []
        for term in self.terms:
            code = encoded[:, term[0]].copy()
            for col in term[1:]:
                code *= self.radices[col]
                code += encoded[:, col]
            parts.append(code)
        return np.column_stack(parts)

    def _scaled(self, matrix):
        return sparse.csr_matrix(matrix.multiply(self.weights))

    def predict_proba(self, rows):
        encoded = self._coded(_clean_categories(rows), False)
        return self.classifier.predict_proba(self._scaled(self.basis.transform(self._crosses(encoded))))
