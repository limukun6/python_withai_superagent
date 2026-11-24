"""Learn a conditional handoff between an equal binary pair and one challenger."""
import math
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

POLICIES = ("hard", "regions", "soft")


def binary(probability, ties):
    return ((probability > .5) | ((probability == .5) & ties)).astype(int)


def inputs(probabilities, pairs):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(pairs), 3) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected three finite, aligned binary probability columns")
    ties = np.asarray([left < right for left, right in pairs], dtype=bool)
    pair = (p[:, 0] + p[:, 1]) / 2
    old, third = binary(pair, ties), binary(p[:, 2], ties)
    unanimous = binary(p[:, 0], ties) == binary(p[:, 1], ties)
    shared = np.column_stack([2 * np.abs(pair - .5), 2 * np.abs(p[:, 2] - .5),
                              np.abs(p[:, 0] - p[:, 1]), unanimous])
    context = []
    for left, right in pairs:
        a, b = (re.findall(r"\w+", s.casefold()) for s in (left, right))
        union = set(a) | set(b)
        context.append([len(set(a) & set(b)) / max(1, len(union)),
                        abs(len(a) - len(b)) / max(1, len(a) + len(b))])
    return {"p": p, "pair": pair, "base": old, "third": third, "conflict": old != third,
            "unanimous": unanimous, "ties": ties, "features": np.column_stack([shared, context])}


def wilson_lower(wins, count):
    z = 1.2815515655446004
    fraction = wins / count
    return (fraction + z*z / (2*count) - z*math.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count))) / (1+z*z/count)


def fit_regions(data, y):
    result = []
    advantage = data["features"][:, 1] - data["features"][:, 0]
    for unanimous in (False, True):
        mask = data["conflict"] & (data["unanimous"] == unanimous)
        best, threshold = 0., None
        if mask.sum() >= 16:
            for low in np.unique(np.quantile(advantage[mask], [0, .25, .5, .75])):
                admitted = mask & (advantage >= low)
                count = int(admitted.sum())
                if count < 16:
                    continue
                wins = int((data["third"][admitted] == y[admitted]).sum())
                gain = count * (2*wilson_lower(wins, count) - 1)
                if gain > best:
                    best, threshold = gain, float(low)
        result.append(threshold)
    return {"kind": "regions", "thresholds": result, "supervised_fits": 1}


def design(data, policy):
    features = data["features"][:, :4] if policy["kind"] == "hard" else data["features"]
    centered = np.clip((features - policy["center"]) / policy["scale"], -5, 5)
    return np.column_stack([np.ones(len(features)), centered])


def fit_policy(name, data, labels):
    if name not in POLICIES:
        raise ValueError("Unknown conditional handoff")
    y = np.asarray(labels)
    if y.shape != data["base"].shape or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Gate labels must follow the same binary rows")
    use = data["conflict"]
    wins = data["third"][use] == y[use]
    if len(wins) < 24 or min(int(wins.sum()), int((~wins).sum())) < 5:
        return {"kind": "baseline", "requested": name, "reason": "insufficient conflicting outcomes", "supervised_fits": 0}
    if name == "regions":
        return fit_regions(data, y)
    features = data["features"][use, :4] if name == "hard" else data["features"][use]
    scale = np.std(features, axis=0)
    policy = {"kind": name, "center": features.mean(axis=0).tolist(),
              "scale": np.where(scale > 1e-9, scale, 1.).tolist(), "supervised_fits": 1}
    x = design(data, policy)[use]
    if name == "hard":
        model = LogisticRegression(C=.25, max_iter=500).fit(x[:, 1:], wins)
        coefficients = np.r_[model.intercept_[0], model.coef_[0]]
    else:
        pair, expert = data["pair"][use], data["p"][use, 2]
        prior = np.zeros(x.shape[1])
        prior[0] = logit(1/3)
        penalty = np.r_[.003, np.full(x.shape[1]-1, .015)]
        def objective(beta):
            weight = expit(x @ beta)
            q = np.clip(pair + weight * (expert - pair), 1e-8, 1 - 1e-8)
            difference = beta - prior
            loss = -np.mean(y[use]*np.log(q) + (1-y[use])*np.log1p(-q)) + .5*np.dot(penalty, difference**2)
            slope = ((q-y[use])/(q*(1-q))) * (expert-pair) * weight * (1-weight)
            return loss, x.T @ slope / len(q) + penalty * difference
        result = minimize(objective, prior, method="L-BFGS-B", jac=True)
        if not result.success:
            raise ValueError("Conditional weight estimation failed")
        coefficients = result.x
    policy["coefficients"] = coefficients.tolist()
    return policy


def predict(policy, data):
    name = policy["kind"]
    if name == "baseline":
        return data["base"].copy()
    if name == "regions":
        advantage = data["features"][:, 1] - data["features"][:, 0]
        take = np.zeros(len(advantage), dtype=bool)
        for i, low in enumerate(policy["thresholds"]):
            if low is not None:
                take |= (data["unanimous"] == bool(i)) & (advantage >= low)
    else:
        weight = expit(design(data, policy) @ policy["coefficients"])
        if name == "soft":
            mixture = (1-weight)*data["pair"] + weight*data["p"][:, 2]
            return np.where(data["conflict"], binary(mixture, data["ties"]), data["base"])
        if name != "hard":
            raise ValueError("Unknown saved policy")
        take = weight > .5
    return np.where(data["conflict"] & take, data["third"], data["base"])


def changes(base, candidate, labels):
    old, new = base == labels, candidate == labels
    gain, loss = int((new & ~old).sum()), int((old & ~new).sum())
    return {"rows": len(old), "base_correct": int(old.sum()), "correct": int(new.sum()),
            "switches": gain+loss, "gained": gain, "lost": loss, "net": gain-loss}


def select(policies, learning, y_learn, confirm, y_confirm):
    minimum = max(2, math.ceil(.002*len(y_confirm)))
    evidence, best, best_key = {}, "baseline", (0, 0)
    for name, policy in policies.items():
        first = changes(learning["base"], predict(policy, learning), y_learn)
        second = changes(confirm["base"], predict(policy, confirm), y_confirm)
        admitted = first["net"] > 0 and second["net"] >= minimum
        evidence[name] = {"learning": first, "confirmation": second, "eligible": admitted}
        key = (second["net"], -second["switches"])
        if admitted and key > best_key:
            best, best_key = name, key
    return {"chosen": best, "required_confirmation_gain": minimum, "evidence": evidence}


def predictions(data, policies, selection):
    views = {"baseline": data["base"], "equal_three": binary(data["p"].mean(axis=1), data["ties"]),
             "third_alone": data["third"]}
    views.update({name: predict(policy, data) for name, policy in policies.items()})
    views["agent"] = views[selection["chosen"]]
    return views
