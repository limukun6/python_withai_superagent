"""Generic feature and estimator alternatives for ordered binary text preferences."""
from functools import partial
import hashlib

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from . import ab_sentence
from .ab_preference_output import CanonicalPreferencePipeline
from .models import Candidate, style_differences
from .nonlinear_text import nonlinear_candidates
from .supervised_text import research_candidates


CATALOG = 'varied'
TOPIC = 'minilm_l12_topic_style_interactions'
BINARY = 'binary_word_direction_naive_bayes'
TOPICS = 8
SAMPLE_LIMIT = 384


def legacy_methods():
    return [c for c in research_candidates() if c.params['C'] == .2] + [
        c for c in nonlinear_candidates() if c.params.get('depth') == 3 and c.params.get('classifier_fits_per_evaluation') == 1]


ESTIMATORS = {c.family: c.params.get('classifier','catboost') for c in legacy_methods()}
ESTIMATORS[BINARY] = 'naive_bayes'
ESTIMATORS[TOPIC] = 'logistic'


class DirectionalWords(TransformerMixin, BaseEstimator):
    def fit(self, rows, y=None):
        self.vectorizer_ = CountVectorizer(ngram_range=(1,2), binary=True, min_df=1, max_features=40000)
        self.vectorizer_.fit([text for pair in rows for text in pair])
        return self

    def transform(self, rows):
        left, right = zip(*rows)
        difference = self.vectorizer_.transform(left) - self.vectorizer_.transform(right)
        return hstack([difference.maximum(0), (-difference).maximum(0)],format='csr')


def fit_bernoulli(rows, y, query):
    labels=np.asarray(y)
    model = CanonicalPreferencePipeline([('features', DirectionalWords()), ('classifier', BernoulliNB(alpha=1.))])
    model.fit([*rows, *((b,a) for a,b in rows)], np.r_[labels,1-labels])
    return model, model.predict(query)


class TopicStyleFeatures(ab_sentence.FamiliarityPreferenceFeatures):
    """Cross symmetric topic coordinates with signed style and training-corpus familiarity differences."""
    def fit(self, rows, y=None):
        super().fit(rows,y)
        pairs = {tuple(sorted((a,b))) for a,b in rows}
        def order(pair):
            return hashlib.sha256(repr(pair).encode('utf-8')).digest(), pair
        subset = sorted(pairs,key=order)[:SAMPLE_LIMIT]
        vectors,tokens = ab_sentence.encode([s for pair in subset for s in pair],deep=True)
        center = .5*(vectors[::2]+vectors[1::2])
        self.components_ = None
        if len(center)>1 and np.any(center != center[0]):
            self.components_ = PCA(n_components=min(TOPICS,len(center)-1,center.shape[1]),svd_solver='full').fit(center)
        self.usage_['input_tokens'] += tokens
        self.usage_['topic_sample_rows'] = len(subset)
        return self

    def semantic_blocks(self, rows, texts):
        vectors,tokens = ab_sentence.encode(texts,deep=True)
        difference = vectors[::2]-vectors[1::2]
        familiarity = self.corpus_.differences(rows)/self.familiarity_scale_
        blocks = [csr_matrix(difference), csr_matrix(familiarity[:,None])]
        if self.components_ is not None:
            model = self.components_
            context = model.transform(.5*(vectors[::2]+vectors[1::2]))
            scale = np.sqrt(np.maximum(model.explained_variance_,1e-6))
            context = np.clip(context/scale,-3,3)/np.sqrt(context.shape[1])
            style = np.column_stack([self.scaler_.transform(style_differences(rows)),familiarity])
            style /= np.sqrt(style.shape[1])
            interaction = (context[:,:,None]*style[:,None,:]).reshape(len(rows),-1)
            blocks.append(csr_matrix(interaction))
        return blocks,tokens


def additions(has_deep_weights, include_legacy_methods=True):
    methods = legacy_methods() if include_legacy_methods else []
    methods.append(Candidate(BINARY,dict(representation='word_ngram_direction_presence',classifier='naive_bayes',
                                         alpha=1.,swap_augmentation=True),fit_bernoulli))
    if has_deep_weights:
        params = dict(encoder=ab_sentence.DEEP_MODEL,revision=ab_sentence.DEEP_REVISION,C=.3,
                      representation='word_char_minilm_l12_training_surprisal_topic_style',topic_dimensions=TOPICS,
                      topic_sample_limit=SAMPLE_LIMIT,swap_augmentation=True)
        methods.append(Candidate(TOPIC,params,partial(ab_sentence.fit_sentence,kind='pair_text',topic_style=True)))
    return methods
