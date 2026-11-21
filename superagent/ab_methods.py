"""Executable early methods for the shared A/B menu. Constructor calls do not train models."""
from functools import partial
from importlib.util import find_spec

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import Candidate
from .ab_symmetric import CHANNELS, ENCODER, candidates as symmetric_candidates
from . import ab_vision


FIELDS = 'supported_field_interactions'
TEXT = 'minilm_text_with_lexical_features'
IMAGE = 'image_gradient_histograms_logistic'
ROUTES = {FIELDS:'field_interactions', TEXT:'minilm_lexical', IMAGE:'gradient_histograms',
          **{name: name for name in CHANNELS}, ENCODER: ENCODER, ab_vision.NAME:'frozen_image_features'}


def _field_tree(rows, labels, query):
    from .field_interactions import InteractionTree
    model = InteractionTree({'pair_limit':3,'backbone_params':{'depth':6,'l2_leaf_reg':3.0}}).fit(rows, labels)
    return model, model.predict(query)


class GradientHistograms(TransformerMixin, BaseEstimator):
    """Pool unsigned edge orientations and brightness within a small fixed spatial grid."""
    def fit(self, rows, y=None):
        values = np.asarray(rows)
        if values.ndim not in (3,4) or min(values.shape[1:3]) < 1:
            raise ValueError('Expected nonempty N,H,W[,C] image arrays')
        self.shape_ = values.shape[1:]
        return self

    def transform(self, rows):
        images = np.asarray(rows)
        if images.shape[1:] != self.shape_:
            raise ValueError('Image shape differs from fitting')
        height, width = images.shape[1:3]
        cy, cx = min(height,4), min(width,4)
        cells = (np.arange(height)[:,None]*cy//height)*cx + np.arange(width)*cx//width
        output = []
        for image in images:
            gray = np.asarray(image,dtype=np.float32)
            if gray.ndim == 3:
                gray = gray[:,:,:3].mean(axis=2)
            dx = np.diff(gray, axis=1, append=gray[:,-1:])
            dy = np.diff(gray, axis=0, append=gray[-1:,:])
            angles = (np.arctan2(dy,dx) % np.pi) * (8/np.pi)
            bins = np.minimum(angles.astype(int),7)
            hist = np.bincount((cells*8+bins).ravel(),weights=np.hypot(dx,dy).ravel(),
                               minlength=cy*cx*8).reshape(cy*cx,8)
            hist /= np.sqrt((hist**2).sum(axis=1,keepdims=True)+1.e-8)
            means = np.bincount(cells.ravel(),weights=gray.ravel())/np.bincount(cells.ravel())
            output.append(np.r_[hist.ravel(),means])
        return np.stack(output)


def _image(rows, labels, query):
    model = Pipeline([('gradient_histograms',GradientHistograms()),('scale',StandardScaler()),
                      ('classifier',LogisticRegression(C=1.,max_iter=2000,random_state=42))])
    model.fit(rows,labels)
    return model,model.predict(query)


def additions(kind):
    if kind=='categorical_tabular' and find_spec('catboost') is not None:
        return [Candidate(FIELDS,dict(pair_limit=3,depth=6,l2_leaf_reg=3.,iterations=600,
                          representation='training-selected categorical field interactions'),_field_tree)]
    if kind=='image':
        methods = [Candidate(IMAGE,dict(grid=4,orientation_bins=8,C=1.,representation='gradient histograms and cell means'),_image)]
        if ab_vision.weights_file() is not None:
            methods.append(Candidate(ab_vision.NAME,dict(encoder=ab_vision.WEIGHT_ID, weight_sha256=ab_vision.WEIGHT_SHA256,
                C=10., representation='L2-normalized frozen 576-dimensional image features'),ab_vision.fit_vision))
        return methods
    if kind in ('single_text','pair_text'):
        from .ab_sentence import MODEL, REVISION, fit_sentence, verified_directory
        encoder_ready = verified_directory() is not None
        extra = symmetric_candidates(include_encoder=encoder_ready) if kind == 'pair_text' else []
        if encoder_ready:
            extra.append(Candidate(TEXT,dict(encoder=MODEL,revision=REVISION,C=.3,representation='word_char_minilm',
                                   swap_augmentation=kind=='pair_text'),partial(fit_sentence,kind=kind)))
        return extra
    return []
