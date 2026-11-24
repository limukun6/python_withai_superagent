"""Nominate and approve a frozen policy on different sets of feedback groups."""
import hashlib

import numpy as np

from tools.feedback_policy_pool import grouped_error
from tools.preference_handoff import binary


def feedback_halves(groups):
    groups = np.asarray(groups)
    ranked = sorted(set(groups), key=lambda g: hashlib.sha256(("lifecycle-confirm-v1/"+str(g)).encode()).digest())
    first_groups = set(ranked[::2])
    first = np.array([g in first_groups for g in groups], dtype=bool)
    return first, ~first


def gains(p, reference, ties, y, groups):
    baseline_correct = binary(reference, ties) == y
    candidate_correct = binary(p, ties) == y
    difference = candidate_correct.astype(int) - baseline_correct.astype(int)
    se = grouped_error(difference, groups)
    return dict(rows=len(y), groups=len(set(groups)), net=int(difference.sum()), gain=float(difference.mean()),
                standard_error=se, lower_one_se=float(difference.mean()-se),
                brier_gain=float(np.mean((reference-y)**2 - (p-y)**2)))


def choose_separately(learn, y_learn, learn_ties, confirm, y_confirm, confirm_ties, groups):
    """Both probability maps are in the same rule order; the incumbent is named baseline."""
    if list(learn) != list(confirm) or "baseline" not in learn:
        raise ValueError("Expected aligned named probability columns and an incumbent")
    for data, y, ties in ((learn, y_learn, learn_ties), (confirm, y_confirm, confirm_ties)):
        if (np.shape(ties) != (len(y),) or not np.isin(y, (0,1)).all()
                or any(np.shape(p) != (len(y),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in data.values())):
            raise ValueError("Nonaligned or invalid labeled policy probabilities")
    if len(groups) != len(y_confirm):
        raise ValueError("Confirmation group count differs")
    first, second = feedback_halves(groups)
    groups = np.asarray(groups)
    role_counts = [dict(rows=int(m.sum()), groups=len(set(groups[m]))) for m in (first, second)]
    chosen, evidence, best = "baseline", {}, (0., 0.)
    if min(c["groups"] for c in role_counts) < 2:
        return dict(nominee=chosen, selected=chosen, halves=role_counts, nominations=evidence, ratification=None)
    for name, p in learn.items():
        if name == "baseline":
            continue
        delta = int((binary(p,learn_ties) == y_learn).sum() - (binary(learn["baseline"],learn_ties) == y_learn).sum())
        gain = gains(confirm[name][first], confirm["baseline"][first], confirm_ties[first], y_confirm[first], groups[first])
        evidence[name] = dict(learning_net=delta, nomination=gain)
        rank = (gain["lower_one_se"], gain["brier_gain"])
        if delta > 0 and gain["lower_one_se"] > 0 and rank > best:
            best, chosen = rank, name
    ratification = None
    selected = "baseline"
    if chosen != "baseline":
        ratification = gains(confirm[chosen][second],confirm["baseline"][second],confirm_ties[second],y_confirm[second],groups[second])
        if ratification["lower_one_se"] > 0:
            selected = chosen
    return dict(nominee=chosen, selected=selected, halves=role_counts, nominations=evidence, ratification=ratification)
