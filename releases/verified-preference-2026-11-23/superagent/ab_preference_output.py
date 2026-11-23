"""A frozen preference pipeline that scores one canonical orientation per input pair."""
import numpy as np
from sklearn.pipeline import Pipeline


class CanonicalPreferencePipeline(Pipeline):
    def _positive_probability(self, rows, **params):
        classes = np.asarray(self.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError('Preference output requires fitted classes 0 and 1')
        ordered, first, equal = [], [], []
        for row in rows:
            if (not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 2
                    or not all(isinstance(text, str) for text in row)):
                raise ValueError('Each preference row needs two strings')
            left, right = row
            ordered.append((left, right) if left <= right else (right, left))
            first.append(left < right)
            equal.append(left == right)
        first = np.asarray(first, dtype=bool)
        equal = np.asarray(equal, dtype=bool)
        if not ordered:
            return np.empty(0, dtype=float), first
        probabilities = np.asarray(super().predict_proba(ordered, **params))
        if (probabilities.shape != (len(ordered), 2) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0.) or np.any(probabilities > 1.)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-8)):
            raise ValueError('Fitted preference model returned invalid probabilities')
        positive = probabilities[:, np.flatnonzero(classes == 1)[0]]
        return np.where(equal, .5, np.where(first, positive, 1. - positive)), first

    def predict_proba(self, X, **params):
        positive, _ = self._positive_probability(X, **params)
        return np.column_stack([positive if label == 1 else 1.-positive for label in self.classes_])

    def predict(self, X, **params):
        positive, first = self._positive_probability(X, **params)
        return ((positive > .5) | ((positive == .5) & first)).astype(self.classes_.dtype)
