"""Bounded challengers for ordered preferences and categorical tables."""
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
import json
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, normalize

from .models import Candidate, StylePairModel, _clean_categories


def _tokens(text):
    tokens=re.findall(r"\w+|[^\w\s]",str(text).casefold())
    return tuple(tokens if len(tokens)<=128 else tokens[:64]+tokens[-64:])


def _edits(pair):
    first,second=map(_tokens,pair)
    if first==second:return {}
    left,right=sorted((first,second))
    orientation=1 if first==left else -1
    counts=Counter()
    for op,start,end,other_start,other_end in SequenceMatcher(None,left,right,autojunk=False).get_opcodes():
        if op=='equal':continue
        for tokens,begin,stop,sign in ((left,start,end,orientation),(right,other_start,other_end,-orientation)):
            for index in range(begin,stop):
                position=min(2,3*index//len(tokens))
                counts[f'{position}:{tokens[index]}']+=sign
                counts[f'edit:{tokens[index]}']+=sign
        if end-start==other_end-other_start==1:
            old,new=left[start],right[other_start]
            counts[repr(('replace',*sorted((old,new))))]+=orientation*(1 if old<new else -1)
    return counts


def edit_features(pairs):
    return normalize(FeatureHasher(n_features=65536,input_type='dict').transform(_edits(pair) for pair in pairs))


class EditPairModel(StylePairModel):
    def transform(self,pairs):
        return hstack([super().transform(pairs),edit_features(pairs)]).tocsr()


class CategoryCrosses(TransformerMixin, BaseEstimator):
    def fit(self,rows,labels=None):
        values=_clean_categories(rows)
        sizes=[len(set(column)) for column in values.T]
        columns=sorted((index for index,size in enumerate(sizes) if 1<size<=64),key=lambda index:(sizes[index],index))
        self.pairs_=list(combinations(columns[:24],2))
        self.width_=values.shape[1]
        return self

    def transform(self,rows):
        values=_clean_categories(rows)
        if values.shape[1]!=self.width_:raise ValueError('Column count changed')
        return np.asarray([list(row)+[json.dumps([row[left],row[right]],ensure_ascii=False)
            for left,right in self.pairs_] for row in values],dtype=object)


def fit_crosses(x_train,y_train,x_search):
    model=make_pipeline(CategoryCrosses(),OneHotEncoder(min_frequency=8,handle_unknown='infrequent_if_exist'),
        LogisticRegression(C=0.1,max_iter=2000,random_state=42))
    model.fit(x_train,y_train)
    return model,model.predict(x_search)


def cross_candidate():
    return Candidate('category_cross_C0.1',{'C':0.1,'minimum_frequency':8,'max_cross_columns':24,
        'max_levels_for_cross':64},fit_crosses)
