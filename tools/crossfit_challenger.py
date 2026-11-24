"""Nominate a representation from residuals and decide a three-member vote with grouped OOF feedback."""
import hashlib
import math

import numpy as np

from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from tools.feedback_policy_pool import grouped_error
from tools.preference_handoff import binary


def ranked_groups(groups, salt):
    return sorted(set(groups), key=lambda g: hashlib.sha256((salt+g).encode()).digest())


def role_indices(groups):
    groups = np.asarray(groups)
    ordered = ranked_groups(groups, "crossfit-challenger-v1/")
    buckets = {g:i%9 for i,g in enumerate(ordered)}
    code = np.asarray([buckets[g] for g in groups])
    return {k:np.flatnonzero(mask) for k,mask in
            (("nominate",code==0),("a",(code>=1)&(code<=4)),("b",code>=5))}


def training_roles(groups, labels):
    groups = np.asarray(groups)
    roles = role_indices(groups)
    if any(len(set(groups[ids])) < 4 or set(labels[ids])!={0,1} for ids in roles.values()):
        return None
    return roles


def group_weights(groups):
    _,ix,counts = np.unique(groups,return_inverse=True,return_counts=True)
    return 1./counts[ix]


def residual_directions(vectors, residual, groups):
    x = np.asarray(vectors,dtype=float)
    r = np.asarray(residual,dtype=float)
    g = np.asarray(groups)
    if x.ndim!=2 or len(x)!=len(r) or len(g)!=len(r) or not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("Invalid residual screening rows")
    ranked = ranked_groups(g,"crossfit-nomination-v1/")
    if len(ranked)<4:
        raise ValueError("Need at least four distinct screen groups")
    first = np.isin(g,ranked[::2])
    x = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    gradients = []
    for mask in (first,~first):
        weight = group_weights(g[mask])
        gradients.append(x[mask].T @ (weight*r[mask]) / weight.sum())
    return np.stack(gradients)


def nominate(scores):
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite nomination score")
    return next((name for name in sorted(scores,key=lambda n:(-scores[n],n)) if scores[name]>0),None)


def components(scores, ties):
    p = np.asarray(scores,dtype=float)
    if p.shape!=(len(ties),3) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Expected three aligned member probabilities")
    disagree = binary(p[:,0],ties)!=binary(p[:,1],ties)
    base = p[:,:2].mean(1)
    return base, p[:,2]-base, disagree


def learn_weights(scores, ties, labels, groups):
    base, delta, disagree = components(scores,ties)
    labels = np.asarray(labels)
    if labels.shape!=base.shape or len(groups)!=len(base) or not np.isin(labels,[0,1]).all():
        raise ValueError("Invalid weight training labels or groups")
    w = group_weights(groups)
    result = []
    for mask in (~disagree,disagree):
        numerator = np.sum(w[mask]*delta[mask]*(labels[mask]-base[mask]))
        denominator = 1.+np.sum(w[mask]*delta[mask]**2)
        result.append(float(np.clip(numerator/denominator,0,1)))
    return result


def combine(scores,ties,weights):
    base, delta, disagree = components(scores,ties)
    factors = np.asarray(weights)
    if factors.shape!=(2,) or not np.isfinite(factors).all() or np.any((factors<0)|(factors>1)):
        raise ValueError("Expected two bounded agreement weights")
    return base + factors[disagree.astype(int)]*delta


def select(feedback):
    learned = {k:learn_weights(**feedback[k]) for k in ("a","b")}
    evidence, rights, all_groups = {}, [], []
    for role,other in (("a","b"),("b","a")):
        d = feedback[role]
        base = d["scores"][:,:2].mean(1)
        predicted = combine(d["scores"],d["ties"],learned[other])
        y = d["labels"]
        gain = (binary(predicted,d["ties"])==y).astype(int) - (binary(base,d["ties"])==y).astype(int)
        evidence[role] = {"rows":len(y),"groups":len(set(d["groups"])),"net":int(gain.sum()),
                          "accuracy_gain":float(gain.mean()),"standard_error":grouped_error(gain,d["groups"]),
                          "brier_gain":float(np.mean((base-y)**2-(predicted-y)**2))}
        rights.append(gain)
        all_groups.extend(d["groups"])
    if sum(e["groups"] for e in evidence.values())!=len(set(all_groups)):
        raise ValueError("OOF panels share a group")
    gain = np.concatenate(rights)
    minimum = max(5,math.ceil(.002*len(gain)))
    se = grouped_error(gain,all_groups)
    accepted = (int(gain.sum())>=minimum and gain.mean()>se and
                all(e["brier_gain"]>0 and e["accuracy_gain"]>=-e["standard_error"] for e in evidence.values()))
    return dict(chosen="agreement_weight" if accepted else "baseline", inner_weights=learned,
                weights=np.mean([learned["a"],learned["b"]],axis=0).tolist(),evidence=evidence,
                minimum_net=minimum,net=int(gain.sum()),pooled_standard_error=se)


class AgreementVote(PreferenceProbabilityVote):
    def __init__(self, models, weights):
        super().__init__("pair_text",list(models),np.array([0,1]))
        self.weights = list(weights)

    def predict_proba(self,rows):
        if not len(rows):
            return np.empty((0,2))
        p = np.column_stack([canonical_probabilities(m,rows)[1][:,1] for m in self.models])
        q = combine(p,np.asarray([a<b for a,b in rows]),self.weights)
        return np.column_stack((1-q,q))
