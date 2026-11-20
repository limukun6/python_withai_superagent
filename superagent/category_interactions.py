"""A supervised linear model of single categories and frequent two-column combinations."""
from itertools import combinations
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from .models import _clean_categories


class CategoryInteractionClassifier:
    def __init__(self,regularization):
        self.regularization=regularization

    def fit(self,rows,labels):
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,dtype=np.int64)
        codes=self.encoder.fit_transform(_clean_categories(rows))
        self.sizes=[len(levels) for levels in self.encoder.categories_]
        self.pairs=[]
        support=max(3,math.ceil(.001*len(codes)))
        for left,right in combinations(range(codes.shape[1]),2):
            keys=codes[:,left]*self.sizes[right]+codes[:,right]
            levels,counts=np.unique(keys,return_counts=True)
            retained=levels[counts>=support]
            if len(retained)>1:
                self.pairs.append((left,right,retained))
        matrix=self.matrix(codes)
        self.classifier=LogisticRegression(C=self.regularization,solver='liblinear' if len(np.unique(labels))==2 else 'lbfgs',
                                           max_iter=500,random_state=42).fit(matrix,labels)
        self.classes_=self.classifier.classes_
        self.training_metadata=dict(single_features=sum(self.sizes),interaction_features=matrix.shape[1]-sum(self.sizes),
                                    column_pairs=len(self.pairs),minimum_support=support,iterations=self.classifier.n_iter_.tolist())
        return self

    def matrix(self,codes):
        row_parts=[]
        column_parts=[]
        offset=0
        for column,size in enumerate(self.sizes):
            known=np.flatnonzero(codes[:,column]>=0)
            row_parts.append(known)
            column_parts.append(offset+codes[known,column])
            offset+=size
        for left,right,levels in self.pairs:
            known=np.flatnonzero((codes[:,left]>=0)&(codes[:,right]>=0))
            keys=codes[known,left]*self.sizes[right]+codes[known,right]
            position=np.searchsorted(levels,keys)
            matched=(position<len(levels))&(levels[np.minimum(position,len(levels)-1)]==keys)
            row_parts.append(known[matched])
            column_parts.append(offset+position[matched])
            offset+=len(levels)
        indices=np.concatenate(column_parts)
        return csr_matrix((np.ones(len(indices)),(np.concatenate(row_parts),indices)),shape=(len(codes),offset))

    def predict_proba(self,rows):
        return self.classifier.predict_proba(self.matrix(self.encoder.transform(_clean_categories(rows))))

    def predict(self,rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]
