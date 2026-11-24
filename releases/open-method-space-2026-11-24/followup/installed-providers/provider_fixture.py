"""Two independent installed-package fixture estimators, using only NumPy during supervised fitting."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MedianPrototypeClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.centers_ = np.stack([np.median(X[y == label], axis=0) for label in self.classes_])
        self.scale_ = np.maximum(np.median(np.abs(X - np.median(X, axis=0)), axis=0), 1e-6)
        return self

    def predict_proba(self, X):
        delta = np.asarray(X)[:, None, :] - self.centers_
        scores = -np.mean(np.abs(delta) / self.scale_, axis=2)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class BinaryThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Two labels required")
        centers = np.stack([X[y == label].mean(axis=0) for label in self.classes_])
        separation = np.abs(centers[0] - centers[1]) / np.maximum(X.std(axis=0), 1e-6)
        self.feature_ = int(separation.argmax())
        self.centers_ = centers[:, self.feature_]
        return self

    def predict(self, X):
        column = np.asarray(X)[:, self.feature_, None]
        return self.classes_[np.abs(column - self.centers_).argmin(axis=1)]
