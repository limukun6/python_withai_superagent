"""Histogram boosting with supervised category partitions and train-fold ordinal codes."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from .models import _clean_categories


class HistogramCategoryClassifier:
    def __init__(self,params):
        self.params=params

    def fit(self,rows,labels):
        self.encoder=OrdinalEncoder(dtype=np.float64,handle_unknown='use_encoded_value',unknown_value=np.nan,max_categories=128)
        x=self.encoder.fit_transform(_clean_categories(rows))
        self.classifier=HistGradientBoostingClassifier(categorical_features=np.ones(x.shape[1],dtype=bool),
            learning_rate=.06,max_iter=self.params['iterations'],max_leaf_nodes=self.params['leaf_nodes'],
            min_samples_leaf=self.params['minimum_leaf'],l2_regularization=self.params['l2'],
            early_stopping=False,random_state=42)
        with threadpool_limits(limits=2):self.classifier.fit(x,labels)
        self.classes_=self.classifier.classes_
        self.training_metadata=dict(backend='hist_gradient_boosting',trees_per_round=int(self.classifier.n_trees_per_iteration_),
                                    rounds=int(self.classifier.n_iter_),columns=x.shape[1],training_rows=len(x))
        return self

    def predict_proba(self,rows):
        x=self.encoder.transform(_clean_categories(rows))
        with threadpool_limits(limits=2):return self.classifier.predict_proba(x)

    def predict(self,rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]
