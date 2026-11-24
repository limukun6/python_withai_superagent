"""Use independent pilot feedback to keep a reference pair on one kind of row."""
import numpy as np

from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from tools.feedback_policy_pool import grouped_error
from tools.preference_handoff import binary, changes

REGIONS = ("agree", "disagree")


def views(scores, ties):
    p = np.asarray(scores, dtype=float)
    if p.shape != (len(ties), 3) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected three aligned binary probabilities")
    agree = binary(p[:, 0], ties) == binary(p[:, 1], ties)
    return agree, p[:, :2].mean(axis=1), p.mean(axis=1)


def reversion(scores, ties, labels, groups, region):
    agree, pair, triple = views(scores, ties)
    mask = agree if region == "agree" else ~agree
    n_groups = len(set(groups[mask]))
    if not mask.any():
        return dict(rows=0, groups=0, switches=0, net=0, brier_gain=None, brier_lower=None)
    y = labels[mask]
    delta = (triple[mask] - y) ** 2 - (pair[mask] - y) ** 2
    gain = float(delta.mean())
    metrics = changes(binary(triple[mask], ties[mask]), binary(pair[mask], ties[mask]), y)
    lower = gain - grouped_error(delta, groups[mask]) if n_groups >= 2 else None
    return {**metrics, "groups": n_groups, "brier_gain": gain, "brier_lower": lower}


def propose(scores, ties, labels, groups):
    results = {region: reversion(scores, ties, labels, groups, region) for region in REGIONS}
    eligible = [name for name, r in results.items() if r["switches"] >= 10 and r["net"] > 0
                and r["brier_lower"] is not None and r["brier_lower"] > 0]
    ranked = sorted(eligible, key=lambda k: (-results[k]["brier_lower"], -results[k]["net"], k))
    return (ranked[0] if ranked else None), results


def confirm(region, scores, ties, labels, groups):
    if region is None:
        return None, None
    if region not in REGIONS:
        raise ValueError("Unknown consensus region")
    evidence = reversion(scores, ties, labels, groups, region)
    passed = evidence["switches"] >= 10 and evidence["net"] >= 2 and evidence["brier_gain"] > 0
    return (region if passed else None), evidence


def adjusted(scores, ties, region):
    agree, pair, triple = views(scores, ties)
    if region is None:
        return triple
    if region not in REGIONS:
        raise ValueError("Unknown consensus region")
    return np.where(agree if region == "agree" else ~agree, pair, triple)


class ConsensusPreferenceVote(PreferenceProbabilityVote):
    def __init__(self, models, region):
        if region not in REGIONS or len(models) != 3:
            raise ValueError("A confirmed region and three ordered fitted members are required")
        super().__init__("pair_text", models, np.array([0, 1]))
        self.region = region

    def predict_proba(self, rows):
        scores = np.column_stack([canonical_probabilities(model, rows)[1][:, 1] for model in self.models])
        ties = np.asarray([a < b for a, b in rows])
        positive = adjusted(scores, ties, self.region)
        return np.column_stack([1 - positive, positive])
