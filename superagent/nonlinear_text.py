"""Optional nonlinear pair-text research using generic shape statistics only."""
from dataclasses import dataclass
from functools import partial

import numpy as np

from .compute_resources import cpu_threads
from .models import Candidate,text_statistics,_word_char_stats_text,probability_with_model
from .supervised_text import _pair_labels


def style_matrix(pairs):
    return np.asarray([np.concatenate([text_statistics(a),text_statistics(b)]) for a,b in pairs],dtype=float).reshape(-1,32)


def _reverse(pairs):return [(b,a) for a,b in pairs]
def _columns(probability):return np.column_stack([1-probability,probability])


@dataclass
class StyleTreePairModel:
    classifier: object
    classifier_fits: int=1

    @property
    def classes_(self):return np.asarray([0,1])

    def predict_proba(self,pairs):
        forward=self.classifier.predict_proba(style_matrix(pairs))[:,1]
        backward=self.classifier.predict_proba(style_matrix(_reverse(pairs)))[:,1]
        return _columns(.5*(forward+1-backward))

    def predict(self,pairs):return self.classes_[np.argmax(self.predict_proba(pairs),axis=1)]


@dataclass
class SymmetricPairModel:
    base: object
    classifier_fits: int=1

    @property
    def classes_(self):return np.asarray([0,1])

    def predict_proba(self,pairs):
        labels,forward=probability_with_model('pair_text',self.base,pairs)
        reverse_labels,backward=probability_with_model('pair_text',self.base,_reverse(pairs))
        if labels.tolist()!=[0,1] or reverse_labels.tolist()!=[0,1]:raise ValueError('Pair probabilities require 0/1 columns')
        return _columns(.5*(forward[:,1]+1-backward[:,1]))

    def predict(self,pairs):return self.classes_[np.argmax(self.predict_proba(pairs),axis=1)]


@dataclass
class FixedStyleBlend:
    linear: SymmetricPairModel
    tree: StyleTreePairModel
    style_weight: float
    classifier_fits: int=2

    @property
    def classes_(self):return np.asarray([0,1])

    def predict_proba(self,pairs):
        if not np.isfinite(self.style_weight) or not 0<self.style_weight<1:raise ValueError('Fixed style weight must be in (0,1)')
        return (1-self.style_weight)*self.linear.predict_proba(pairs)+self.style_weight*self.tree.predict_proba(pairs)

    def predict(self,pairs):return self.classes_[np.argmax(self.predict_proba(pairs),axis=1)]


def fit_style_tree(x_train,y_train,x_search,depth=3):
    from catboost import CatBoostClassifier
    y=_pair_labels(y_train,len(x_train));reverse=_reverse(x_train)
    matrix=np.vstack([style_matrix(x_train),style_matrix(reverse)]);labels=np.concatenate([y,1-y])
    classifier=CatBoostClassifier(iterations=300,depth=depth,learning_rate=.05,l2_leaf_reg=5.,
        loss_function='Logloss',random_seed=42,thread_count=cpu_threads(2),verbose=False,allow_writing_files=False)
    classifier.fit(matrix,labels)
    if np.asarray(classifier.classes_).tolist()!=[0,1]:raise ValueError('Unexpected fitted class order')
    model=StyleTreePairModel(classifier);return model,model.predict(x_search)


def fit_symmetric_linear(x_train,y_train,x_search):
    y=_pair_labels(y_train,len(x_train))
    base,_=_word_char_stats_text(x_train,y,x_train[:1],c=1.)
    model=SymmetricPairModel(base);return model,model.predict(x_search)


def fit_style_blend(x_train,y_train,x_search,depth=3,weight=.2):
    if not np.isfinite(weight) or not 0<weight<1:raise ValueError('Fixed style weight must be in (0,1)')
    y=_pair_labels(y_train,len(x_train))
    linear,_=fit_symmetric_linear(x_train,y,x_train[:1])
    tree,_=fit_style_tree(x_train,y,x_train[:1],depth)
    model=FixedStyleBlend(linear,tree,weight);return model,model.predict(x_search)


def nonlinear_candidates():
    common={'iterations':300,'learning_rate':.05,'l2_leaf_reg':5.,'swap_augmentation':True,'symmetric_probability':True}
    tree=[Candidate(f'style_tree_depth{depth}',{**common,'depth':depth,'representation':'generic_text_statistics',
        'classifier_fits_per_evaluation':1},partial(fit_style_tree,depth=depth)) for depth in (3,5)]
    blends=[Candidate(f'style_blend_depth{depth}_weight{weight}',{**common,'depth':depth,'style_weight':weight,'C':1.,
        'representation':'word_char_stats_and_statistic_tree','classifier_fits_per_evaluation':2},
        partial(fit_style_blend,depth=depth,weight=weight)) for depth in (3,5) for weight in (.2,.4)]
    symmetric=Candidate('symmetric_stats_C1.0',{'C':1.,'representation':'word_char_stats','swap_augmentation':True,
        'symmetric_probability':True,'classifier_fits_per_evaluation':1},fit_symmetric_linear)
    return [*tree,*blends,symmetric]
