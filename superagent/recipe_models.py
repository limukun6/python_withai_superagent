"""Validated text recipes and train-fitted latent semantic representations.

Latent semantic analysis here is TF-IDF/SVD, not a pretrained language model.
"""
from functools import partial
import hashlib
import json
import math
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


def validate_recipe(spec):
    required = {'kind', 'analyzer', 'representation', 'classifier', 'C', 'dimensions'}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError('Recipe must have exactly the declared fields')
    if spec['kind'] not in ('single_text', 'pair_text') or spec['analyzer'] not in ('word', 'char_wb'):
        raise ValueError('Unsupported recipe modality or analyzer')
    if spec['representation'] not in ('lexical', 'latent', 'hybrid') or spec['classifier'] not in ('logistic', 'linear_svm'):
        raise ValueError('Unsupported representation or classifier')
    if type(spec['C']) not in (int, float) or not math.isfinite(spec['C']) or not .05 <= spec['C'] <= 10.:
        raise ValueError('Recipe C must be finite in [0.05,10]')
    if type(spec['dimensions']) is not int or spec['dimensions'] not in (32, 64):
        raise ValueError('Recipe dimension must be 32 or 64')
    return json.loads(json.dumps(spec, allow_nan=False))


class RecipeTextModel:
    def __init__(self, spec):
        self.spec = validate_recipe(spec)

    def _documents(self, inputs):
        return [str(s) for pair in inputs for s in pair] if self.spec['kind'] == 'pair_text' else list(inputs)

    def _features(self, inputs):
        sparse = self.vectorizer.transform(self._documents(inputs))
        if self.spec['representation'] == 'lexical':
            matrix = sparse
        else:
            latent = self.normalizer.transform(self.reducer.transform(sparse))
            matrix = csr_matrix(latent) if self.spec['representation'] == 'latent' else hstack([sparse, csr_matrix(latent)]).tocsr()
        if self.spec['kind'] == 'pair_text':
            matrix = matrix[0::2] - matrix[1::2]
        return matrix

    def fit(self, inputs, labels):
        word = self.spec['analyzer'] == 'word'
        self.vectorizer = TfidfVectorizer(analyzer=self.spec['analyzer'], ngram_range=(1, 2) if word else (3, 5),
                                         min_df=1, sublinear_tf=True, max_features=20000)
        train = self.vectorizer.fit_transform(self._documents(inputs))
        if self.spec['representation'] != 'lexical':
            n = min(self.spec['dimensions'], train.shape[1] - 1, train.shape[0] - 1)
            if n < 1:
                raise ValueError('Too little training vocabulary for latent features')
            self.reducer = TruncatedSVD(n_components=n, n_iter=5, random_state=42).fit(train)
            self.normalizer = Normalizer().fit(self.reducer.transform(train))
        matrix = self._features(inputs)
        y = np.asarray(labels)
        if self.spec['kind'] == 'pair_text':
            if not set(y.tolist()) <= {0, 1}:
                raise ValueError('Pair recipe needs binary labels')
            matrix = vstack([matrix, -matrix]); y = np.concatenate([y, 1 - y])
        self.classifier = (LogisticRegression(C=self.spec['C'], max_iter=2000, random_state=42)
                           if self.spec['classifier'] == 'logistic' else
                           LinearSVC(C=self.spec['C'], max_iter=5000, random_state=42))
        self.classifier.fit(matrix, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, inputs):
        return self.classifier.predict(self._features(inputs))

    def predict_proba(self, inputs):
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError('Linear SVM recipe has no probability output')
        return self.classifier.predict_proba(self._features(inputs))


def fit_recipe(spec, x_train, y_train, x_search):
    model = RecipeTextModel(spec).fit(x_train, y_train)
    return model, model.predict(x_search)


def materialize_recipe(spec):
    from .models import Candidate
    spec = validate_recipe(spec)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    route = 'recipe_' + '_'.join(spec[k] for k in ('kind', 'representation', 'analyzer', 'classifier'))
    return Candidate(route + '_' + digest, spec, partial(fit_recipe, spec))


def semantic_candidates(kind):
    if kind not in ('pair_text', 'single_text'):
        return []
    return [materialize_recipe({'kind': kind, 'analyzer': 'word', 'representation': representation,
                               'classifier': 'logistic', 'C': c, 'dimensions': 32})
            for representation in ('latent', 'hybrid') for c in (.3, 1.)]
