"""A fixed random Fourier map appended to the original standardized numeric features."""
import hashlib

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted


class NumericKernelFeatures(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        x = check_array(x, dtype=np.float64)
        self.n_features_in_ = x.shape[1]

        def rank(index):
            vector = np.ascontiguousarray(x[index], dtype='<f8')
            return hashlib.sha256(vector.tobytes()).digest()

        sample = sorted(range(len(x)), key=rank)[:384]
        distances = pdist(x[sample], metric='sqeuclidean')
        positive = distances[distances > 0]
        self.median_squared_distance_ = float(np.median(positive)) if positive.size else 0.0
        gamma = np.log(2.0) / self.median_squared_distance_ if positive.size else 1.0
        self.scale_ = float(np.sqrt(np.einsum('ij,ij->', x, x) / len(x)))
        self.map_ = RBFSampler(gamma=gamma, n_components=512, random_state=20261123).fit(x)
        return self

    def transform(self, x):
        check_is_fitted(self, 'map_')
        x = check_array(x, dtype=np.float64)
        return np.column_stack((x, self.scale_ * self.map_.transform(x)))


def add_kernel(baseline):
    original = clone(baseline)
    return Pipeline([*original.steps[:-1], ('kernel', NumericKernelFeatures()), original.steps[-1]])
