"""Select individual experts or static / example-dependent probability fusion using disjoint feedback."""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from tools.preference_handoff import binary, changes, inputs


def panel(scores, pairs):
    p = np.asarray(scores, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(pairs) or p.shape[1] < 3 or not len(p):
        raise ValueError("Require matched rows and at least three expert probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid expert probabilities")
    context = inputs(p[:, :3], pairs)
    return {"scores": p, "base": context["base"], "ties": context["ties"], "text": context["features"][:, 4:]}


def features(data):
    p = data["scores"]
    base = p[:, :2].mean(1, keepdims=True)
    repeat = np.ones_like(p)
    return np.stack((2*np.abs(p-.5), 2*np.abs(base-.5)*repeat, np.abs(p-base), (2*p-1)*(2*base-1),
                     data["text"][:, :1]*repeat, data["text"][:, 1:]*repeat), axis=2)


def designs(data, rule):
    p = data["scores"]
    intercept = np.ones((*p.shape, 1))
    if rule["kind"] == "constant":
        return intercept
    standardized = (features(data) - rule["center"]) / rule["scale"]
    return np.concatenate((intercept, np.clip(standardized, -5, 5)), axis=2)


def fit_mixture(kind, data, labels):
    p = data["scores"]
    y = np.asarray(labels)
    if kind not in ("constant", "contextual") or y.shape != (len(p),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid fusion family or feedback")
    n = p.shape[1]
    prior = np.full(n, .2/(n-2))
    prior[:2] = .4
    rule = {"kind": kind, "prior": prior.tolist()}
    if kind == "contextual":
        z = features(data)
        scale = z.std(0)
        rule.update(center=z.mean(0).tolist(), scale=np.where(scale > 1e-8, scale, 1.).tolist())
    x = designs(data, rule)
    shape = x.shape[1:]
    penalty = np.full((1, shape[1]), .02)
    penalty[0, 0] = .004

    def objective(flat):
        beta = flat.reshape(shape)
        w = softmax(np.log(prior) + np.einsum("imk,mk->im", x, beta), axis=1)
        prob = np.clip(np.sum(w*p, axis=1), 1e-9, 1-1e-9)
        loss = -np.mean(y*np.log(prob) + (1-y)*np.log1p(-prob)) + .5*np.sum(penalty*beta**2)
        residual = ((prob-y)/(prob*(1-prob)))[:, None] * w * (p-prob[:, None])
        gradient = np.einsum("im,imk->mk", residual, x) / len(y) + penalty*beta
        return loss, gradient.ravel()

    result = minimize(objective, np.zeros(np.prod(shape)), method="L-BFGS-B", jac=True,
                      options={"maxiter": 300, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"Fusion did not converge: {result.message}")
    rule["coefficients"] = result.x.reshape(shape).tolist()
    return rule


def weights(rule, data):
    p = data["scores"]
    if rule["kind"] == "subset":
        w = np.zeros(p.shape[1])
        w[rule["indices"]] = 1/len(rule["indices"])
        return np.broadcast_to(w, p.shape)
    x = designs(data, rule)
    return softmax(np.log(rule["prior"]) + np.einsum("imk,mk->im", x, rule["coefficients"]), axis=1)


def probabilities(rule, data):
    p = data["scores"]
    if rule["kind"] == "subset":
        return p[:, rule["indices"]].mean(1)
    return (p * weights(rule, data)).sum(1)


def subset(indices):
    return {"kind": "subset", "indices": list(indices)}


def shortlist(data, labels, size):
    best, best_key = None, None
    for ix in combinations(range(data["scores"].shape[1]), size):
        rule = subset(ix)
        changed = changes(data["base"], binary(probabilities(rule, data), data["ties"]), labels)
        rank = (changed["net"], -changed["switches"])
        if best_key is None or rank > best_key:
            best, best_key = rule, rank
    return best


def evaluate_rule(rule, data, labels):
    return changes(data["base"], binary(probabilities(rule, data), data["ties"]), labels)


def choose(names, learning, y_learn, confirm, y_confirm):
    if len(names) != len(set(names)) or learning["scores"].shape[1] != len(names) or confirm["scores"].shape[1] != len(names):
        raise ValueError("Unmatched expert identities in learning and confirmation")
    rules = {"baseline": subset((0, 1)),
             "single": shortlist(learning, y_learn, 1), "pair": shortlist(learning, y_learn, 2),
             "constant": fit_mixture("constant", learning, y_learn), "contextual": fit_mixture("contextual", learning, y_learn)}
    minimum = max(2, math.ceil(.002*len(y_confirm)))
    chosen, best_key, evidence = "baseline", (0, 0), {}
    for name, rule in rules.items():
        train = evaluate_rule(rule, learning, y_learn)
        fresh = evaluate_rule(rule, confirm, y_confirm)
        eligible = train["net"] > 0 and fresh["net"] >= minimum
        evidence[name] = {"learning": train, "confirmation": fresh, "eligible": eligible}
        key = (fresh["net"], -fresh["switches"])
        if eligible and key > best_key:
            chosen, best_key = name, key
    return {"members": list(names), "chosen": chosen, "minimum_confirmation_gain": minimum,
            "rules": rules, "evidence": evidence, "continuous_fits": 2, "supervised_shortlists": 2}


def outputs(decision, data):
    p, ties = data["scores"], data["ties"]
    columns = {name: binary(p[:, i], ties) for i, name in enumerate(decision["members"])}
    columns.update({name: binary(probabilities(rule, data), ties) for name, rule in decision["rules"].items()})
    columns["agent"] = columns[decision["chosen"]]
    return columns


@dataclass
class LearnedPreferenceVote(PreferenceProbabilityVote):
    rule: dict

    def predict_proba(self, rows):
        if not len(rows):
            return np.empty((0, 2))
        p = np.column_stack([canonical_probabilities(m, rows)[1][:, 1] for m in self.models])
        positive = probabilities(self.rule, panel(p, rows))
        return np.column_stack([1-positive, positive])


def delivery(models, decision):
    rule = decision["rules"][decision["chosen"]]
    if rule["kind"] == "subset":
        return PreferenceProbabilityVote("pair_text", [models[i] for i in rule["indices"]], np.array([0, 1]))
    return LearnedPreferenceVote("pair_text", list(models), np.array([0, 1]), rule)
