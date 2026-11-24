"""Skip a third probability evaluation only for already decided binary labels."""
import numpy as np

from .ab_complement import PreferenceProbabilityVote, canonical_probabilities


class BoundedPreferenceVote(PreferenceProbabilityVote):
    def __init__(self, models):
        if len(models) != 3:
            raise ValueError("The bounds require an equal vote of three binary components")
        super().__init__("pair_text", models, np.array([0, 1]))

    def predict_with_counts(self, rows):
        n = len(rows)
        counts = dict(rows=n, certified=0, third_rows=0, full_batch_rechecks=0)
        if not n:
            return np.empty(0, dtype=int), counts
        first, second = [canonical_probabilities(m, rows)[1][:, 1] for m in self.models[:2]]
        # Preserve the ordinary mean's rounding for lower-precision components.
        tol = 64 * np.finfo(np.float64).eps
        total = first.astype(np.float64) + second
        stable = (total < .5 - tol) | (total > 1.5 + tol)
        if first.dtype != np.float64 or second.dtype != np.float64:
            stable[:] = False
        result = (total > 1.5).astype(int)
        pending = np.flatnonzero(~stable)
        counts["certified"] = n - len(pending)
        if not len(pending):
            return result, counts
        inputs = rows if len(pending) == n else [rows[i] for i in pending]
        third = canonical_probabilities(self.models[2], inputs)[1][:, 1]
        counts["third_rows"] = len(pending)
        mean = np.mean([first[pending], second[pending], third], axis=0)
        # A different batch layout may perturb a neural encoder's last few digits.
        if len(pending) < n and np.any(np.abs(mean - .5) <= .00002):
            third = canonical_probabilities(self.models[2], rows)[1][pending, 1]
            mean = np.mean([first[pending], second[pending], third], axis=0)
            counts["third_rows"] += n
            counts["full_batch_rechecks"] = 1
        ties = np.asarray([a < b for a, b in inputs], dtype=bool)
        result[pending] = ((mean > .5) | ((mean == .5) & ties)).astype(int)
        return result, counts

    def predict(self, rows):
        return self.predict_with_counts(rows)[0]

    def predict_proba(self, rows):
        if not len(rows):
            return np.empty((0, 2))
        return super().predict_proba(rows)
