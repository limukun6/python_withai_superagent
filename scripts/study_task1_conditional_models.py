"""Research candidates for tables with recurring attribute blocks and changing context."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from superagent.attribute_alignment import aligned_columns, comparable_fields, fitted_routes, information
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.models import _clean_categories, _tabular_catboost
from superagent.splits import sample_groups


class ResponseMap:
    """Estimate contextual effects relative to a field's wider response distribution."""

    def __init__(self, x, target, fields, gate):
        self.fields, self.gate = fields, gate
        groups = x[:,gate] if gate is not None else np.full(len(x),"")
        global_p = (target.sum()+1)/(len(x)+2)
        self.fallback = float(global_p)
        self.maps = {}
        marginal = {}
        for j in fields:
            for v in np.unique(x[:,j]):
                use = x[:,j] == v
                marginal[j,v] = (target[use].sum()+16*global_p)/(use.sum()+16)
        for key in np.unique(groups):
            mask = groups == key
            y = target[mask]
            prior = (y.sum()+24*global_p)/(len(y)+24)
            effect, support = {}, {}
            scores = []
            for j in fields:
                values = x[mask,j]
                scores.append(information(values,y,2))
                for value in np.unique(values):
                    match = values==value
                    size = int(match.sum())
                    # A context main effect and a field main effect form the small-cell prior.
                    offset = logit(prior)+logit(marginal[j,value])-logit(global_p)
                    smoothed = (y[match].sum()+16/(1+np.exp(-offset)))/(size+16)
                    effect[j,value] = float(logit(smoothed)-logit(prior))
                    support[j,value] = size/(size+16)
            weights = softmax(np.asarray(scores)*np.sqrt(len(y)))
            self.maps[key] = (float(prior), weights, effect, support)

    def transform(self, x):
        arr = np.zeros((len(x),3))
        arr[:,0] = logit(self.fallback)
        for i,row in enumerate(x):
            key = row[self.gate] if self.gate is not None else ""
            table = self.maps.get(key)
            if table is None:
                continue
            prior, weights, effect, support = table
            keys = [(j,row[j]) for j in self.fields]
            e = np.array([effect.get(k,0.) for k in keys])
            mean = float(weights@e)
            arr[i] = logit(prior)+mean, np.sqrt(weights@np.square(e-mean)), weights@[support.get(k,0.) for k in keys]
        return arr


class PosteriorAlignmentTree(ContextTokenTree):
    def __init__(self):
        super().__init__({"pair_limit":3})

    def representation(self, x, target, families):
        routes = fitted_routes(x,target,families,self.context)
        return routes, [ResponseMap(x,target,f,g) for f,(g,_,_) in zip(families,routes,strict=True)]

    def extra(self, x, state):
        routes, maps = state
        return aligned_columns(x,routes), np.column_stack([m.transform(x) for m in maps])

    def matrix(self, x, cats, dense):
        nominal = np.column_stack((ContextTokenTree.transform(self,x),cats))
        return pd.concat((pd.DataFrame(nominal),pd.DataFrame(dense)),axis=1,ignore_index=True), nominal.shape[1]

    def fit(self, rows, y):
        x = _clean_categories(rows)
        self.profile,self.context = blocks(x)
        families = comparable_fields(x,self.profile)
        self.classes_,target = np.unique(y,return_inverse=True)
        if len(self.classes_) != 2 or not families:
            raise ValueError("Binary labels and a family of comparable fields are required")
        self.pairs = choose_pairs(rank_pairs(x,y),3)
        groups = sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
        cats = np.empty((len(x),2*len(families)),dtype=object)
        dense = np.empty((len(x),3*len(families)),dtype=float)
        for fit,held in GroupKFold(3).split(x,target,groups):
            if set(groups[fit]).intersection(groups[held]):
                raise RuntimeError("Overlapping groups in fitted features")
            state = self.representation(x[fit],target[fit],families)
            cats[held],dense[held] = self.extra(x[held],state)
        self.state = self.representation(x,target,families)
        features,width = self.matrix(x,cats,dense)
        self.classifier = _tabular_catboost(y,6,3.)
        self.classifier.set_params(thread_count=4)
        self.classifier.fit(features,y,cat_features=list(range(width)))
        self.audit = dict(families=len(families),family_sizes=list(map(len,families)),
                          supervised_map_constructions=4*len(families),held_group_overlap=0)
        return self

    def transform(self, rows):
        x = _clean_categories(rows)
        return self.matrix(x,*self.extra(x,self.state))[0]


class CrossedLogistic:
    """Partially pool profile and context interactions through block-specific shrinkage."""

    def columns(self, x):
        profile = tokens(x[:,self.profile])
        context = tokens(x[:,self.context])
        columns = [x[:,j] for j in self.fields] + [profile,context]
        scale = [1.]*len(self.fields) + [.65,.55]
        for i,j in self.pairs:
            columns.append(tokens(x[:,[i,j]]))
            scale.append(.35)
        for j in self.context:
            columns.append(tokens(np.column_stack((profile,x[:,j]))))
            scale.append(.48)
        return np.column_stack(columns),scale

    def fit(self, rows, labels):
        x = _clean_categories(rows)
        self.profile,self.context = blocks(x)
        self.fields = sorted(set(self.profile)|set(self.context))
        counts = {j:len(set(x[:,j])) for j in self.fields}
        self.pairs = [(a,b) for a,b in combinations(self.fields,2) if (a in self.context or b in self.context)
                      and counts[a]*counts[b] <= len(x)//5]
        columns,scale = self.columns(x)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        design = self.encoder.fit_transform(columns)
        self.scales = np.repeat(scale,[len(v) for v in self.encoder.categories_])
        self.classifier = LogisticRegression(C=1.,max_iter=500,solver="lbfgs",tol=1e-6)
        self.classifier.fit(design.multiply(self.scales),labels)
        self.classes_ = self.classifier.classes_
        self.audit = dict(pair_terms=len(self.pairs),parameter_count=design.shape[1],supervised_map_constructions=0)
        return self

    def predict_proba(self, rows):
        cols,_ = self.columns(_clean_categories(rows))
        return self.classifier.predict_proba(self.encoder.transform(cols).multiply(self.scales))
