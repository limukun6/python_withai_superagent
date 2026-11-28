"""A population-level candidate for categorical tables with recurring blocks."""
import numpy as np
from sklearn.model_selection import GroupKFold

from superagent.attribute_alignment import aligned_columns, comparable_fields, fitted_routes
from superagent.field_interactions import InteractionTree, choose_pairs, rank_pairs
from superagent.joint_context import blocks, tokens
from superagent.models import _clean_categories, _tabular_catboost
from superagent.profile_history import discover_columns


def novel_queries(fit, queries):
    table = _clean_categories(fit)
    cols = discover_columns(table)
    if not cols:
        return np.zeros(len(queries), dtype=bool)
    known = set(tokens(table[:, cols]))
    return np.asarray([k not in known for k in tokens(_clean_categories(queries)[:, cols])])


class PopulationTree(InteractionTree):
    def __init__(self):
        super().__init__({"pair_limit": 3})

    def base_features(self, table):
        return np.column_stack((InteractionTree.transform(self, table), tokens(table[:, self.context])))

    def fit(self, rows, labels):
        x = _clean_categories(rows)
        y = np.asarray(labels)
        self.profile, self.context = blocks(x)
        group_keys = tokens(x[:, self.profile])
        self.known = set(group_keys)
        _, groups, sizes = np.unique(group_keys, return_inverse=True, return_counts=True)
        self.pairs = choose_pairs(rank_pairs(x, y), self.params["pair_limit"])
        families = comparable_fields(x, self.profile)
        self.routes = []
        features = self.base_features(x)
        if families:
            extra = np.empty((len(x), 2 * len(families)), dtype=object)
            for fit, held in GroupKFold(3).split(x, y, groups):
                routes = fitted_routes(x[fit], y[fit], families, self.context)
                extra[held] = aligned_columns(x[held], routes)
            self.routes = fitted_routes(x, y, families, self.context)
            features = np.column_stack((features, extra))
        weights = 1 / np.sqrt(sizes[groups])
        weights = np.clip(weights / weights.mean(), .35, 3.)
        self.classifier = _tabular_catboost(y, 6, 3.)
        self.classifier.set_params(thread_count=4)
        self.classifier.fit(features, y, sample_weight=weights/weights.mean(),
                            cat_features=list(range(features.shape[1])))
        self.classes_ = self.classifier.classes_
        self.audit = dict(training_profiles=len(sizes), learnt_interactions=len(self.pairs),
                          attribute_families=len(families), supervised_route_estimations=4 if families else 0)
        return self

    def novel(self, rows):
        keys = tokens(_clean_categories(rows)[:, self.profile])
        return np.asarray([k not in self.known for k in keys])

    def transform(self, rows):
        x = _clean_categories(rows)
        base = self.base_features(x)
        return np.column_stack((base, aligned_columns(x, self.routes))) if self.routes else base
