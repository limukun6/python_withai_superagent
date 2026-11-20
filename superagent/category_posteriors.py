"""Learn smoothed category and pair effects without using a row group's own labels."""
from itertools import combinations

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

from .models import _clean_categories, _tabular_catboost


class CategoryPosteriors:
    def __init__(self, include_pairs, prior_count=20):
        self.include_pairs = include_pairs
        self.prior_count = prior_count

    def _codes(self, rows):
        # Code zero is reserved for an unseen level.
        return self.encoder.transform(_clean_categories(rows, full_marker=True)).astype(np.int64) + 1

    def _keys(self, codes, fields):
        if len(fields) == 1:
            return codes[:, fields[0]]
        left, right = fields
        return codes[:, left] * self.sizes[right] + codes[:, right]

    def _count_tables(self, codes, labels):
        n_classes = len(self.classes_)
        prior = (np.bincount(labels, minlength=n_classes) + 1) / (len(labels) + n_classes)
        tables = {}
        for fields in self.fields:
            size = int(np.prod([self.sizes[i] for i in fields]))
            combined = self._keys(codes, fields) * n_classes + labels
            tables[fields] = np.bincount(combined, minlength=size*n_classes).reshape(size, n_classes)
        return prior, tables

    def _transform(self, codes, evidence):
        prior, tables = evidence
        probabilities = {}
        for fields in self.fields:
            counts = tables[fields][self._keys(codes, fields)]
            if len(fields) == 1:
                backoff = prior
            else:
                left, right = fields
                backoff = (probabilities[(left,)] + probabilities[(right,)]) / 2
            probabilities[fields] = (counts + self.prior_count*backoff) / (counts.sum(axis=1, keepdims=True) + self.prior_count)
        return np.column_stack([matrix[:, 1:] for matrix in probabilities.values()]).astype(np.float32)

    def fit_transform(self, rows, labels):
        raw = _clean_categories(rows, full_marker=True)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(raw)
        codes = self.encoder.transform(raw).astype(np.int64) + 1
        self.classes_, label_codes = np.unique(labels, return_inverse=True)
        self.sizes = [len(values)+1 for values in self.encoder.categories_]
        columns = [i for i, size in enumerate(self.sizes) if 3 <= size <= 257]
        if not columns:
            raise ValueError("No repeated categorical fields")
        self.fields = [(i,) for i in columns]
        if self.include_pairs:
            pairs = [(a, b) for a,b in combinations(columns, 2) if (self.sizes[a]-1)*(self.sizes[b]-1) <= len(rows)//4]
            # Prefer repeated pairs with more resolution, using only feature cardinalities.
            pairs.sort(key=lambda pair: (-((self.sizes[pair[0]]-1)*(self.sizes[pair[1]]-1)), pair))
            self.fields.extend(pairs[:24])
        _, groups = np.unique(codes, axis=0, return_inverse=True)
        fold_count = min(5, int(groups.max())+1)
        if fold_count < 2:
            raise ValueError("At least two distinct feature groups are needed")
        output = np.empty((len(rows), len(self.fields)*(len(self.classes_)-1)), dtype=np.float32)
        for train, held in GroupKFold(n_splits=fold_count).split(codes, groups=groups):
            local = self._count_tables(codes[train], label_codes[train])
            output[held] = self._transform(codes[held], local)
        self.fitted = self._count_tables(codes, label_codes)
        return output

    def transform(self, rows):
        return self._transform(self._codes(rows), self.fitted)


class PosteriorCategoryClassifier:
    def __init__(self, params):
        self.params = params

    def fit(self, rows, labels):
        self.encoder = CategoryPosteriors(self.params["include_pairs"], self.params["prior_count"])
        encoded = self.encoder.fit_transform(rows, labels)
        raw = _clean_categories(rows, full_marker=True).astype(object)
        params = self.params["backbone_params"]
        self.classifier = _tabular_catboost(labels, params["depth"], params["l2_leaf_reg"])
        self.classifier.fit(np.column_stack((raw, encoded)), labels, cat_features=list(range(raw.shape[1])))
        self.classes_ = self.classifier.classes_
        self.training_metadata = dict(backend="catboost_cross_fitted_category_posteriors",
                                      fields=len(self.encoder.fields), numeric_features=encoded.shape[1],
                                      prior_count=self.encoder.prior_count, training_rows=len(rows))
        return self

    def predict_proba(self, rows):
        raw = _clean_categories(rows, full_marker=True).astype(object)
        return self.classifier.predict_proba(np.column_stack((raw, self.encoder.transform(rows))))

    def predict(self, rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]
