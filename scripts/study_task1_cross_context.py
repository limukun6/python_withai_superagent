"""Learn signed transfer between contexts of a recurring profile using only its other training responses."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.feature_extraction import DictVectorizer

from scripts.study_task1_adaptive_memory import cached_probability
from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens
from superagent.models import _clean_categories

PLAN = ROOT / "research/task1_feedback_extension_1127/cross_context_transfer.json"


class ContextHistory:
    def __init__(self, x, y):
        self.profile, self.context = blocks(x)
        self.x = x
        self.rows = defaultdict(list)
        profile = tokens(x[:, self.profile])
        items = tokens(x[:, self.context])
        self.exact = tokens(x)
        profile_count, item_count, joint_count = Counter(profile), Counter(items), Counter(zip(profile, items))
        profile_yes, item_yes, joint_yes = Counter(), Counter(), Counter()
        for i, (p, c) in enumerate(zip(profile, items, strict=True)):
            self.rows[p].append(i)
            profile_yes[p] += int(y[i])
            item_yes[c] += int(y[i])
            joint_yes[p, c] += int(y[i])
        positives = int(y.sum())
        self.residual = []
        for i, (p, c) in enumerate(zip(profile, items, strict=True)):
            global_rate = (positives - profile_yes[p] + 1) / (len(y) - profile_count[p] + 2)
            other_count = item_count[c] - joint_count[p, c]
            expected = (item_yes[c] - joint_yes[p, c] + 16 * global_rate) / (other_count + 16)
            self.residual.append(float(y[i]) - expected)

    def features(self, x):
        data, supports = [], []
        profiles, exacts = tokens(x[:, self.profile]), tokens(x)
        for row, key, exact in zip(x, profiles, exacts, strict=True):
            previous = [j for j in self.rows.get(key, ()) if self.exact[j] != exact]
            record = defaultdict(float)
            for j in previous:
                source = self.x[j]
                residual = self.residual[j]
                record["all"] += residual
                matched = []
                for pos in self.context:
                    if row[pos] == source[pos]:
                        matched.append(pos)
                        record[f"match_{pos}"] += residual
                    lo, hi = sorted((source[pos], row[pos]))
                    record[f"pair_{pos}_{len(lo)}:{lo}{hi}"] += residual
                for a, b in combinations(matched, 2):
                    record[f"both_{a}_{b}"] += residual
            scale = np.sqrt(len(previous) + 3)
            data.append({k:v / scale for k,v in record.items()})
            supports.append(len(previous))
        return data, supports


def fit_offset(features, y, prior, positive):
    offset = logit(np.clip(prior, .001, .999))
    strength = 24.
    def objective(coef):
        eta = offset + features @ coef
        loss = np.logaddexp(0., eta).sum() - y @ eta + .5 * strength * (coef @ coef)
        gradient = features.T @ (expit(eta) - y) + strength * coef
        return loss, gradient
    result = minimize(objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B",
                      bounds=[(0., None)] * features.shape[1] if positive else None,
                      options=dict(maxiter=300, ftol=1e-11))
    require(bool(result.success) and np.isfinite(result.x).all(), "Unconverged context transfer coefficients")
    return result.x, dict(iterations=int(result.nit), negative=int(np.sum(result.x < -1e-5)),
                          positive=int(np.sum(result.x > 1e-5)))


def train(out):
    inputs = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", inputs)
    x = _clean_categories(fitting["x"])
    classes, y = np.unique(fitting["y"], return_inverse=True)
    require(len(classes) == 2, "Binary offset needed")
    panels = [_clean_categories(bound(out, r + "_features.joblib", inputs)) for r in ROLES]
    query = np.concatenate(panels)
    source = out / "adaptive_memory_components"
    teacher = read(source / "teachers.json")
    existing = out / "distill_teachers"
    require(teacher["borrowed"] == sha(existing / "sealed.json") and read(existing / "sealed.json")["inputs"] == sha(out / "inputs.json")
            and sha(existing / "out_of_fold.npy") == read(existing / "sealed.json")["predictions"]
            and teacher["probabilities"] == sha(source / "aligned_out_of_fold.npy"), "Unbound teachers")
    old = np.load(existing / "out_of_fold.npy", allow_pickle=False)
    aligned = np.load(source / "aligned_out_of_fold.npy", allow_pickle=False)
    history = ContextHistory(x, y)
    training, fit_support = history.features(x)
    testing, query_support = history.features(query)
    vectorizer = DictVectorizer()
    train_features, query_features = vectorizer.fit_transform(training), vectorizer.transform(testing)
    area = out / "cross_context_history"
    area.mkdir(exist_ok=False)
    save(area / "sealed.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), teacher=sha(source / "teachers.json"),
         fitting_rows=len(y), fit_median_history=float(np.median(fit_support)), query_median_history=float(np.median(query_support)),
         without_query_history=int(np.sum(np.asarray(query_support) == 0)), keys=len(vectorizer.vocabulary_),
         rate_memory_fits=1, external_labels=0))
    for base, prior in (("pair", (aligned + old[:, 1]) / 2), ("anchor", old[:, 0])):
        chunks = []
        for role in ROLES:
            if base == "anchor":
                chunks.append(cached_probability(out, "trial_anchor", role, classes))
            else:
                a = cached_probability(out, "trial_attribute_alignment", role, classes)
                b = cached_probability(out, "trial_pretrained_icl", role, classes)
                chunks.append((a + b) / 2)
        start = np.concatenate(chunks)
        for rule in ("signed", "positive"):
            coef, diagnostics = fit_offset(train_features, y, prior, positive=rule == "positive")
            change = np.clip(query_features @ coef, -1.25, 1.25)
            p = expit(logit(np.clip(start, .001, .999)) + change)
            p[change == 0.] = start[change == 0.]
            require(np.isfinite(p).all(), "Nonfinite cross-context prediction")
            folder = out / f"crosscontext_{base}_{rule}"
            folder.mkdir(exist_ok=False)
            signatures = {}
            for role, positive_p in zip(ROLES, np.split(p, np.cumsum([len(v) for v in panels])[:-1]), strict=True):
                f = folder / (role + ".npy")
                np.save(f, np.column_stack((1 - positive_p, positive_p)), allow_pickle=False)
                signatures[role] = sha(f)
            np.save(folder / "coefficient.npy", coef, allow_pickle=False)
            save(folder / "sealed.json", dict(protocol=sha(PLAN), memory=sha(area / "sealed.json"),
                 classes=classes.tolist(), predictions=signatures, coefficients=sha(folder / "coefficient.npy"),
                 supervised_offset_fits=1, fit=diagnostics, api_requests=0))


def grade(out):
    elect(out, ("pair_signed", "pair_positive"), "crosscontext", PLAN)
    info = read(out / "inputs.json")
    classes = np.asarray(read(out / "trial_anchor/sealed.json")["classes"])
    comparisons = {}
    for role in ROLES:
        y = bound(out, role + "_answers.joblib", info)
        base = cached_probability(out, "trial_anchor", role, classes)
        old = classes[(base > .5).astype(int)] == y
        comparisons[role] = {}
        for rule in ("positive", "signed"):
            p = cached_probability(out, "crosscontext_anchor_" + rule, role, classes)
            comparisons[role][rule] = paired(classes[(p > .5).astype(int)] == y, old)
    save(out / "crosscontext_election/anchors.json", comparisons)
    print(comparisons, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    (train if args.action == "train" else grade)(args.out)
