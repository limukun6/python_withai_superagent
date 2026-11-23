"""Executable early methods for the shared A/B menu. Constructor calls do not train models."""
from functools import partial
from importlib.util import find_spec
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .models import Candidate
from .ab_symmetric import CHANNELS, ENCODER, UnorderedPairFeatures, candidates as symmetric_candidates
from . import ab_text_recipes, ab_vision, symmetric_text


FIELDS = 'supported_field_interactions'
TEXT = 'minilm_text_with_lexical_features'
TEXT_KERNEL = 'minilm_text_with_nonlinear_lexical_features'
TEXT_DEEP = 'minilm_l12_text_with_lexical_features'
TEXT_EDIT = 'minilm_l12_with_edit_context'
TEXT_FAMILIARITY = 'minilm_l12_with_corpus_familiarity'
TRANSFER_METHODS={'deep-e5':('e5','e5_small_text_with_lexical_features'),
                  'deep-bge':('bge','bge_small_text_with_lexical_features')}
PREFERENCE_CATALOG_ENV = 'SUPERAGENT_AB_PREFERENCE_CATALOG'
JOINT_PREFERENCE = 'deep-e5-familiarity'
IMAGE = 'image_gradient_histograms_logistic'
RELATION = 'ordered_text_pair_interactions'
ROUTES = {FIELDS:'field_interactions', TEXT:'minilm_lexical', TEXT_KERNEL:'minilm_nonlinear_lexical', IMAGE:'gradient_histograms',
          RELATION:'ordered_text_interactions', **{name: name for name in CHANNELS}, ENCODER: ENCODER,
          ab_vision.NAME:'frozen_image_features'}
ROUTES.update({f'symmetric_{representation}_C{c}':f'symmetric_{representation}'
               for representation in ('word','word_char') for c in (.3,3.)})
ROUTES.update(ab_text_recipes.ROUTES)
ROUTES.update({TEXT_DEEP:'minilm_l12_lexical',TEXT_EDIT:'minilm_l12_edit_context'})
ROUTES[TEXT_FAMILIARITY] = 'minilm_l12_corpus_familiarity'
ROUTES.update({name:variant+'_lexical' for variant,name in TRANSFER_METHODS.values()})


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


def _ordered_relation(rows, labels, query):
    from .general_models import TextTablePreprocessor
    from .schema import TextTableRows

    if not isinstance(rows, TextTableRows) or rows.shape[1] != 2 or rows.text_columns != (0, 1):
        raise ValueError('Ordered text interactions require two declared text columns')
    features = FeatureUnion([('ordered', TextTablePreprocessor()),
                             ('overlap', UnorderedPairFeatures(('word', 'char'), bins=32768))])
    model = Pipeline([('features', features),
                      ('classifier', LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    model.fit(rows, labels)
    return model, model.predict(query)


def additions(kind):
    if kind == 'text_tabular':
        return [Candidate(RELATION, dict(C=1.0, hash_bins=32768, max_iter=2000,
                          representation='ordered columns with word and character pair interactions'), _ordered_relation)]
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
        from .ab_sentence import KERNEL_DIMENSIONS, KERNEL_SAMPLE_LIMIT, KERNEL_SEED, MODEL, REVISION, fit_sentence, verified_directory
        encoder_ready = verified_directory() is not None
        extra = []
        if kind == 'pair_text':
            extra.extend(symmetric_text.candidates())
            extra.extend(symmetric_candidates(include_encoder=encoder_ready))
        if encoder_ready:
            extra.append(Candidate(TEXT,dict(encoder=MODEL,revision=REVISION,C=.3,representation='word_char_minilm',
                                   swap_augmentation=kind=='pair_text'),partial(fit_sentence,kind=kind)))
        catalog=os.environ.get(PREFERENCE_CATALOG_ENV,'standard')
        if kind=='pair_text' and catalog in ('deep','deep-edit','deep-familiarity',JOINT_PREFERENCE,*TRANSFER_METHODS) and verified_directory(deep=True) is not None:
            from .ab_sentence import DEEP_MODEL, DEEP_REVISION
            params=dict(encoder=DEEP_MODEL,revision=DEEP_REVISION,C=.3,
                        representation='word_char_minilm_l12',swap_augmentation=True)
            extra.append(Candidate(TEXT_DEEP,params,partial(fit_sentence,kind=kind,deep=True)))
            if catalog=='deep-edit':
                extra.append(Candidate(TEXT_EDIT,dict(params,representation='word_char_minilm_l12_edit_residual'),
                                       partial(fit_sentence,kind=kind,deep=True,edit_context=True)))
            if catalog in ('deep-familiarity',JOINT_PREFERENCE):
                extra.append(Candidate(TEXT_FAMILIARITY,dict(params,representation='word_char_minilm_l12_training_surprisal'),
                                       partial(fit_sentence,kind=kind,familiarity=True)))
        transfer_catalog = 'deep-e5' if kind == 'pair_text' and catalog == JOINT_PREFERENCE else catalog
        if transfer_catalog in TRANSFER_METHODS:
            from .ab_sentence import ALTERNATIVES,verified_files
            variant,name=TRANSFER_METHODS[transfer_catalog]
            spec=ALTERNATIVES[variant]
            if verified_files(spec['env'],spec['files']) is not None:
                params=dict(encoder=spec['model'],revision=spec['revision'],C=.3,representation='word_char_'+variant,
                            pooling=spec['pool'],input_prefix=spec['prefix'],swap_augmentation=kind=='pair_text')
                extra.append(Candidate(name,params,partial(fit_sentence,kind=kind,alternative=variant)))
        if encoder_ready:
            extra.append(Candidate(TEXT_KERNEL,dict(encoder=MODEL,revision=REVISION,C=.3,representation='word_char_minilm_rbf',
                                   kernel_components=KERNEL_DIMENSIONS,bandwidth_sample_limit=KERNEL_SAMPLE_LIMIT,
                                   kernel_seed=KERNEL_SEED,swap_augmentation=kind=='pair_text'),
                                   partial(fit_sentence,kind=kind,nonlinear=True)))
        extra.extend(ab_text_recipes.candidates(kind))
        return extra
    return []
