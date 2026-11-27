"""Test whether frequent categorical conjunctions add independent probability evidence."""
import argparse
import itertools
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens

PLAN = ROOT / "research/task1_feedback_extension_1127/frequent_conjunctions.json"
METHODS = ("pairs", "triples")


def representation(train, query):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    left = encoder.fit_transform(train) + 1
    right = encoder.transform(query) + 1
    keep, partitions = [], set()
    for j, col in enumerate(left.T):
        levels, first, inverse = np.unique(col, return_index=True, return_inverse=True)
        signature = np.argsort(np.argsort(first))[inverse].tobytes()
        if len(levels) > 1 and signature not in partitions:
            keep.append(j)
            partitions.add(signature)
    augmented = []
    try:
        families = blocks(train)
    except ValueError:
        families = ()
    for columns in families:
        categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        values = categorical.fit_transform(np.asarray(tokens(train[:, columns]))[:, None])[:, 0] + 1
        other = categorical.transform(np.asarray(tokens(query[:, columns]))[:, None])[:, 0] + 1
        augmented.append((values, other))
    return left[:, keep], right[:, keep], augmented


def design(train, query, order):
    left, right, extra = representation(train, query)
    support = {1: 2, 2: 10, 3: 20}
    matrix_rows, matrix_cols, matrix_values = [], [], []
    offset = 0
    n = len(left)

    def add(code, minimum, scale):
        nonlocal offset
        values, counts = np.unique(code[:n], return_counts=True)
        allowed = values[(counts >= minimum) & (counts < .97 * n)]
        if not len(allowed):
            return
        ix = np.searchsorted(allowed, code)
        valid = np.minimum(ix, len(allowed) - 1)
        rows = np.flatnonzero(allowed[valid] == code)
        matrix_rows.append(rows.astype(np.int32))
        matrix_cols.append((offset + valid[rows]).astype(np.int32))
        matrix_values.append(np.full(len(rows), scale, dtype=np.float32))
        offset += len(allowed)

    all_rows = np.concatenate((left, right))
    radices = left.max(axis=0) + 1
    for size in range(1, order + 1):
        scale = 1 / math.sqrt(math.comb(left.shape[1], size))
        for group in itertools.combinations(range(left.shape[1]), size):
            code = np.zeros(len(all_rows), dtype=np.int64)
            for j in group:
                code = code * radices[j] + all_rows[:, j]
            add(code, support[size], scale)
    for a, b in extra:
        add(np.concatenate((a, b)), support[1], 1.)
    compressed = sparse.coo_matrix((np.concatenate(matrix_values), (np.concatenate(matrix_rows),
        np.concatenate(matrix_cols))), shape=(len(all_rows), offset)).tocsr()
    return compressed[:n], compressed[n:]


def run(out):
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    features = [bound(out, r + "_features.joblib", info) for r in ROLES]
    for name, order in zip(METHODS, (2, 3)):
        folder = out / ("rules_" + name)
        folder.mkdir(exist_ok=False)
        started = time.monotonic()
        train, query = design(fitting["x"], np.concatenate(features), order)
        model = LogisticRegression(C=8., solver="liblinear", dual=True, tol=.0002, max_iter=500, random_state=3468)
        model.fit(train, fitting["y"])
        probabilities = model.predict_proba(query)
        require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1.), "Invalid predictions")
        signatures = {}
        for role, p in zip(ROLES, np.split(probabilities, np.cumsum([len(f) for f in features])[:-1])):
            path = folder / (role + ".npy")
            np.save(path, p, allow_pickle=False)
            signatures[role] = sha(path)
        record = dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), classes=model.classes_.tolist(),
                      model_fits=1, training_rows=train.shape[0], features=train.shape[1],
                      nonzero=train.nnz, predictions=signatures, seconds=round(time.monotonic()-started, 2), api_requests=0)
        save(folder / "sealed.json", record)
        print(name, train.shape, train.nnz, record["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "elect"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "run":
        run(args.out)
    else:
        elect(args.out, METHODS, "rules", PLAN)
