"""Optional train-only pair-text methods; these are local research candidates.

Log-count weighting is inspired by Wang & Manning (ACL 2012), equation 2:
https://aclanthology.org/P12-2018.pdf
Our preferred/nonpreferred pair adaptation with logistic regression is not a
reproduction of their NBSVM classifier or their reported results.
"""
from dataclasses import dataclass
from functools import partial
import warnings

import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .models import Candidate,StylePairModel,_difference_features,style_differences


@dataclass
class NBPairModel:
    vectorizer: object
    log_count_ratio: np.ndarray
    classifier: object

    @property
    def classes_(self):return self.classifier.classes_

    def transform(self,pairs):
        left=self.vectorizer.transform([a for a,b in pairs])
        right=self.vectorizer.transform([b for a,b in pairs])
        return (left-right).multiply(self.log_count_ratio).tocsr()

    def predict(self,pairs):return self.classifier.predict(self.transform(pairs))
    def predict_proba(self,pairs):return self.classifier.predict_proba(self.transform(pairs))
    def decision_function(self,pairs):return self.classifier.decision_function(self.transform(pairs))


def _pair_labels(labels,n):
    y=np.asarray(labels)
    if y.ndim!=1 or len(y)!=n or set(y.tolist())!={0,1}:
        raise ValueError('Pair training needs both binary classes, with 1 meaning left is preferred')
    return y


def fit_nb_word(x_train,y_train,x_search,c=0.2):
    y=_pair_labels(y_train,len(x_train))
    vectorizer=CountVectorizer(ngram_range=(1,2),binary=True,min_df=2,max_features=50_000)
    left=[a for a,b in x_train];right=[b for a,b in x_train]
    vectorizer.fit(left+right)
    a,b=vectorizer.transform(left),vectorizer.transform(right)
    preferred=np.asarray(a[y==1].sum(axis=0)+b[y==0].sum(axis=0)).ravel()+1.
    nonpreferred=np.asarray(b[y==1].sum(axis=0)+a[y==0].sum(axis=0)).ravel()+1.
    ratio=np.log(preferred/preferred.sum())-np.log(nonpreferred/nonpreferred.sum())
    matrix=(a-b).multiply(ratio).tocsr()
    classifier=LogisticRegression(C=c,fit_intercept=False,max_iter=2000,random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        classifier.fit(vstack([matrix,-matrix]),np.concatenate([y,1-y]))
    model=NBPairModel(vectorizer,ratio,classifier)
    return model,model.predict(x_search)


def fit_style_svm(x_train,y_train,x_search,c=0.2):
    y=_pair_labels(y_train,len(x_train))
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=40_000)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=40_000)
    a=_difference_features(word,x_train,True);b=_difference_features(char,x_train,True)
    raw=style_differences(x_train)
    scaler=StandardScaler(with_mean=False).fit(np.vstack([raw,-raw]))
    styles=scaler.transform(raw)/np.sqrt(raw.shape[1])
    matrix=hstack([a,b,csr_matrix(styles)]).tocsr()
    classifier=LinearSVC(C=c,loss='squared_hinge',fit_intercept=False,max_iter=5000,dual='auto',random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        classifier.fit(vstack([matrix,-matrix]),np.concatenate([y,1-y]))
    model=StylePairModel(word,char,scaler,classifier)
    return model,model.predict(x_search)


def research_candidates():
    return [Candidate(f'{name}_C{c}',{'C':c,'representation':name,'swap_augmentation':True,
        'fit_intercept':False,'supervised_feature_weighting':name=='nb_word',
        'classifier':'logistic' if name=='nb_word' else 'linear_svm',
        'convergence_warning_policy':'record_as_failed_fit'},partial(function,c=c))
        for name,function in [('nb_word',fit_nb_word),('word_char_stats_svm',fit_style_svm)] for c in (.05,.2,1.)]
