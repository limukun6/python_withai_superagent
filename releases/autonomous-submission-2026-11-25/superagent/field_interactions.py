"""Find categorical interactions that improve held-out class probabilities in both directions."""
from itertools import combinations

import numpy as np

from .models import _clean_categories, _tabular_catboost


def rank_pairs(rows, labels):
    x = _clean_categories(rows)
    classes, target = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(classes) > 16 or len(x) < 200 or not 2 <= x.shape[1] <= 40:
        return []
    codes = []
    for column in x.T:
        levels, encoded = np.unique(column, return_inverse=True)
        codes.append((len(levels), encoded))
    _, groups = np.unique(np.column_stack([c for _, c in codes]), axis=0, return_inverse=True)
    group_count = int(groups.max()) + 1
    if group_count < 20:
        return []
    order = np.random.default_rng(31837).permutation(group_count)
    half = order[groups] < group_count // 2
    if min(half.sum(), (~half).sum()) < 50:
        return []
    n_classes = len(classes)

    def counts(code, size, mask):
        return np.bincount(code[mask] * n_classes + target[mask],
                           minlength=size * n_classes).reshape(size, n_classes)

    results = []
    for a, b in combinations(range(len(codes)), 2):
        na, left = codes[a]
        nb, right = codes[b]
        if min(na, nb) < 2 or max(na, nb) > 64 or na * nb > len(x) // 4:
            continue
        pair = left * nb + right
        effects = []
        for train in (half, ~half):
            held = ~train
            prior = np.bincount(target[train], minlength=n_classes) + 1.
            prior /= prior.sum()
            pa = counts(left, na, train) + 16. * prior
            pb = counts(right, nb, train) + 16. * prior
            # The additive log-odds model is the no-interaction reference.
            additive = (pa / pa.sum(1, keepdims=True))[:, None, :] * (pb / pb.sum(1, keepdims=True))[None, :, :] / prior
            additive = additive.reshape(-1, n_classes)
            additive /= additive.sum(1, keepdims=True)
            table = counts(pair, na * nb, train) + 24. * additive
            table /= table.sum(1, keepdims=True)
            key = (pair[held], target[held])
            effects.append(float(np.mean(np.log(table[key]) - np.log(additive[key]))))
        results.append(dict(fields=[a, b], gains=effects, support=min(effects)))
    return sorted(results, key=lambda r: (-r['support'], r['fields']))


def choose_pairs(ranking, limit):
    return [r['fields'] for r in ranking if r['support'] >= .001][:limit]


class InteractionTree:
    def __init__(self, params):
        self.params = params

    def transform(self, rows):
        x = _clean_categories(rows)
        extras = [[f'{len(v)}:{v}{w}' for v, w in zip(x[:, a], x[:, b], strict=True)] for a, b in self.pairs]
        return np.column_stack((x, *extras)) if extras else x

    def fit(self, rows, labels):
        ranking = rank_pairs(rows, labels)
        self.pairs = choose_pairs(ranking, self.params['pair_limit'])
        self.training_metadata = dict(selected_field_pairs=self.pairs, strongest=ranking[:6],
                                      selection='minimum held-out log-probability gain in two training-group halves')
        base = self.params['backbone_params']
        self.classifier = _tabular_catboost(labels, base['depth'], base['l2_leaf_reg'])
        features = self.transform(rows)
        self.classifier.fit(features, labels, cat_features=list(range(features.shape[1])))
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, rows):
        return self.classifier.predict_proba(self.transform(rows))

    def predict(self, rows):
        return np.asarray(self.classifier.predict(self.transform(rows))).reshape(-1)
