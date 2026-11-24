"""Use grouped probability feedback to pool candidate policies, then confirm the proposed change."""
import numpy as np

from tools.preference_handoff import binary, changes
from tools.preference_portfolio import probabilities


def grouped_error(values, groups):
    x = np.asarray(values, dtype=float)
    _, ix = np.unique(groups, return_inverse=True)
    n = len(ix)
    count = int(ix.max()) + 1
    if len(x) != n or count < 2:
        raise ValueError("At least two aligned independent groups are required")
    centered = np.bincount(ix, weights=x-x.mean())
    return float(np.sqrt(np.dot(centered, centered)*count/(count-1))/n)


def signals(rule, base_rule, data, labels, groups):
    y = np.asarray(labels)
    base = probabilities(base_rule, data)
    p = probabilities(rule, data)
    gain = (base-y)**2 - (p-y)**2
    right = (binary(p, data["ties"]) == y).astype(float) - (data["base"] == y).astype(float)
    return {"brier_gain": float(gain.mean()), "brier_se": grouped_error(gain,groups),
            "accuracy_gain": float(right.mean()), "accuracy_se": grouped_error(right,groups)}


def pooled_probability(weight, rules, data):
    return sum(w*probabilities(rules[name], data) for name, w in weight.items() if w > 0)


def evidence_pool(choice, learning, y_learn, learn_groups, confirm, y_confirm, confirm_groups):
    rules = choice["rules"]
    support, errors, evidence = {}, [], {}
    for name, rule in rules.items():
        if name == "baseline":
            continue
        first = signals(rule, rules["baseline"], learning, y_learn, learn_groups)
        second = signals(rule, rules["baseline"], confirm, y_confirm, confirm_groups)
        strength = min(first["brier_gain"], second["brier_gain"] - second["brier_se"])
        magnitude = max(0., strength) if all(s["accuracy_gain"] >= -s["accuracy_se"] for s in (first,second)) else 0.
        if magnitude:
            support[name] = magnitude
            errors.append(second["brier_se"])
        evidence[name] = {"learning": first, "confirmation": second, "support": magnitude}
    support["baseline"] = float(np.mean(errors)) if errors else 1.
    total = sum(support.values())
    weight = {name: value/total for name, value in support.items()}
    first = changes(learning["base"], binary(pooled_probability(weight, rules, learning), learning["ties"]), y_learn)
    second = changes(confirm["base"], binary(pooled_probability(weight, rules, confirm), confirm["ties"]), y_confirm)
    accepted = first["net"] > 0 and second["net"] >= choice["minimum_confirmation_gain"]
    return {"proposed_weights": weight, "deployed_weights": weight if accepted else {"baseline": 1.},
            "adopted": accepted, "proposed_learning": first, "proposed_confirmation": second, "support": evidence}
