"""Train-only response histories for repeated latent profiles in categorical tables."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .field_interactions import InteractionTree, choose_pairs, rank_pairs
from .models import _clean_categories, _tabular_catboost


def discover_columns(rows):
    """Find a recurring joint configuration unusually frequent under independent columns."""
    x = _clean_categories(rows)
    n = len(x)
    if n < 200:
        return ()
    columns = []
    partitions = set()
    for i, col in enumerate(x.T):
        codes, levels = pd.factorize(col, sort=False)
        if not 2 <= len(levels) <= 128 or codes.tobytes() in partitions:
            continue
        columns.append((i, codes, len(levels)))
        partitions.add(codes.tobytes())
    # Bound the subset lattice on wide tables. Keep high-cardinality fields first.
    columns = sorted(sorted(columns, key=lambda c: -c[2])[:20], key=lambda c: c[0])
    if len(columns) < 3:
        return ()
    rng = np.random.default_rng(56473)
    left = rng.integers(n, size=400_000)
    right = (left + rng.integers(1, n, size=len(left))) % n
    matches = np.zeros(len(left), dtype=np.int32)
    size = 1 << len(columns)
    independence_bits = np.zeros(size)
    for j, (_, codes, _) in enumerate(columns):
        bit = 1 << j
        agree = codes[left] == codes[right]
        matches += agree.astype(np.int32) * bit
        independence_bits[bit:2*bit] = independence_bits[:bit] - np.log2(max(1 / len(left), agree.mean()))
    counts = np.bincount(matches, minlength=size)
    for j in range(len(columns)):
        bit = 1 << j
        blocks = counts.reshape(-1, 2 * bit)
        blocks[:, :bit] += blocks[:, bit:]
    log_prob = np.log2(np.maximum(1, counts) / len(left))
    score = independence_bits + 1.55 * log_prob
    score[counts < max(40, len(left) * 5 / n)] = -np.inf
    # The lift must survive the key-cardinality penalty and cover repeated rows.
    for idx in np.argsort(score)[-32:][::-1]:
        if score[idx] < 2:
            break
        key = tuple(c for j, (c, _, _) in enumerate(columns) if idx & (1 << j))
        if len(key) < 2:
            continue
        frequencies = pd.DataFrame(x[:, key]).value_counts(sort=False)
        if frequencies[frequencies >= 4].sum() / n >= .65:
            return key
    return ()


def profile_keys(x, columns):
    return [tuple(row) for row in x[:, list(columns)]]


class History:
    def __init__(self, x, y, columns, context):
        self.columns = columns
        self.context = context
        keys = profile_keys(x, columns)
        self.groups = {}
        self.prior = (y.sum() + 1) / (len(y) + 2)
        for k, key in enumerate(keys):
            self.groups.setdefault(key, []).append(k)
        self.samples = x[:, [j for j, _ in context]]
        self.labels = np.asarray(y)

    def transform(self, x):
        out = []
        for row, key in zip(x, profile_keys(x, self.columns), strict=True):
            ids = self.groups.get(key, [])
            labels = self.labels[ids]
            samples = self.samples[ids]
            count = len(labels)
            rate = (labels.sum() + 3 * self.prior) / (count + 3)
            features = [count, rate]
            agreements = []
            for pos, (col, levels) in enumerate(self.context):
                match = samples[:, pos] == row[col]
                agreements.append(match)
                for mask in (match, *(samples[:, pos] == level for level in levels)):
                    support = int(mask.sum())
                    features.extend((support, (labels[mask].sum() + 2 * rate) / (support + 2)))
            if agreements and count:
                similarity = np.sum(agreements, axis=0)
                nearest = labels[similarity == np.max(similarity)]
                features.extend((np.max(similarity), len(nearest), (nearest.sum() + 2 * rate) / (len(nearest) + 2)))
            else:
                features.extend((0, 0, rate))
            out.append(features)
        return np.asarray(out, dtype=np.float32)


class ProfileTree(InteractionTree):
    def fit(self, rows, labels):
        x = _clean_categories(rows)
        self.classes_, y = np.unique(labels, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError('Response-history candidate requires two classes')
        self.columns = discover_columns(x)
        if not self.columns:
            raise ValueError('No supported repeated categorical profile found')
        self.context = []
        for j in range(x.shape[1]):
            levels = np.unique(x[:, j])
            if j not in self.columns and 2 <= len(levels) <= 64:
                self.context.append((j, levels.tolist() if len(levels) <= 8 else []))
        exact = pd.factorize(pd.Series(profile_keys(x, tuple(range(x.shape[1])))))[0]
        folds = GroupKFold(n_splits=min(8, len(set(exact))))
        extra = None
        for train, held in folds.split(x, groups=exact):
            part = History(x[train], y[train], self.columns, self.context).transform(x[held])
            if extra is None:
                extra = np.empty((len(x), part.shape[1]), dtype=part.dtype)
            extra[held] = part
        self.history = History(x, y, self.columns, self.context)
        self.pairs = choose_pairs(rank_pairs(x, labels), self.params['pair_limit'])
        cats = super().transform(x)
        features = pd.concat([pd.DataFrame(cats), pd.DataFrame(extra)], axis=1, ignore_index=True)
        base = self.params['backbone_params']
        self.classifier = _tabular_catboost(labels, base['depth'], base['l2_leaf_reg'])
        self.classifier.set_params(thread_count=6)
        self.classifier.fit(features, labels, cat_features=list(range(cats.shape[1])))
        self.classes_ = self.classifier.classes_
        return self

    def transform(self, rows):
        x = _clean_categories(rows)
        cats = super().transform(x)
        extra = self.history.transform(x)
        return pd.concat([pd.DataFrame(cats), pd.DataFrame(extra)], axis=1, ignore_index=True)
