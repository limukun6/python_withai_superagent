"""Estimate partially pooled categorical interactions using a Bayesian factorization machine."""
import argparse
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens

PLAN = ROOT / "research/task1_feedback_extension_1127/bayesian_interactions.json"
METHODS = ("fields", "recurrence")


def fields(x, query, enriched):
    chosen, partitions = [], set()
    for j in range(x.shape[1]):
        _, first, inverse = np.unique(x[:, j], return_index=True, return_inverse=True)
        signature = np.argsort(np.argsort(first))[inverse].tobytes()
        if len(first) > 1 and signature not in partitions:
            partitions.add(signature)
            chosen.append(j)
    base, test = x[:, chosen], query[:, chosen]
    if not enriched:
        return base, test
    try:
        profile, context = blocks(x)
    except ValueError:
        return base, test
    keys = np.asarray(tokens(x[:, profile]))
    other = np.asarray(tokens(query[:, profile]))
    train_fields = [base, keys, tokens(x[:, context])]
    predict_fields = [test, other, tokens(query[:, context])]
    for j in context:
        values = np.asarray(tokens(np.column_stack((keys, x[:, j]))))
        levels, counts = np.unique(values, return_counts=True)
        frequent = levels[counts >= 4]
        if len(frequent) < 2:
            continue
        test_values = np.asarray(tokens(np.column_stack((other, query[:, j]))))
        train_fields.append(np.where(np.isin(values, frequent), values, ""))
        predict_fields.append(np.where(np.isin(test_values, frequent), test_values, ""))
    return np.column_stack(train_fields), np.column_stack(predict_fields)


def run(out):
    from myfm import MyFMClassifier
    require(version("myfm") == "0.4.0", "Unexpected sampler version")
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    panels = [bound(out, r + "_features.joblib", info) for r in ROLES]
    classes, labels = np.unique(fitting["y"], return_inverse=True)
    require(len(classes) == 2, "A probit classifier needs two labels")
    for name in METHODS:
        folder = out / ("factor_" + name)
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        train, query = fields(fitting["x"], np.concatenate(panels), name == "recurrence")
        encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
        design = encoder.fit_transform(train).tocsr()
        sizes = [len(v) for v in encoder.categories_]
        model = MyFMClassifier(rank=8, random_seed=19731)
        model.fit(design, labels, group_shapes=sizes, n_iter=160, n_kept_samples=96)
        query_design = encoder.transform(query).tocsr()
        positive = np.asarray(model.predict_proba(query_design, n_workers=2)).reshape(-1)
        require(len(positive) == len(query) and np.isfinite(positive).all()
                and np.min(positive) >= 0 and np.max(positive) <= 1, "Invalid posterior predictions")
        signatures = {}
        probabilities = np.column_stack((1-positive, positive))
        for role, output in zip(ROLES, np.split(probabilities, np.cumsum([len(r) for r in panels])[:-1])):
            path = folder / (role + ".npy")
            np.save(path, output, allow_pickle=False)
            signatures[role] = sha(path)
        report = dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), classes=classes.tolist(), fits=1,
                      fields=len(sizes), features=design.shape[1], draws=96, seconds=round(time.monotonic()-start, 2),
                      predictions=signatures, api_requests=0)
        save(folder / "sealed.json", report)
        print(name, report["fields"], report["features"], report["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "elect"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "run":
        run(args.out)
    else:
        elect(args.out, METHODS, "factor", PLAN)
